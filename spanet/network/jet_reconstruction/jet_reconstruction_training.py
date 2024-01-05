from typing import Tuple, Dict, List
import numpy as np
import torch
from torch import Tensor, Size
from torch.nn import functional as F

from spanet.options import Options
from spanet.dataset.types import Batch, Source, AssignmentTargets
from spanet.dataset.regressions import regression_loss
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork
from spanet.network.utilities.divergence_losses import assignment_cross_entropy_loss, jensen_shannon_divergence

def numpy_tensor_array(tensor_list):
    output = np.empty(len(tensor_list), dtype=object)
    output[:] = tensor_list

    return output


class JetReconstructionTraining(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetReconstructionTraining, self).__init__(options, torch_script)

        self.log_clip = torch.log(10 * torch.scalar_tensor(torch.finfo(torch.float32).eps)).item()

        self.event_particle_names = list(self.training_dataset.event_info.product_particles.keys())
        self.product_particle_names = {
            particle: self.training_dataset.event_info.product_particles[particle][0]
            for particle in self.event_particle_names
        }

    def particle_symmetric_loss(self, assignment: Tensor, detection: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        assignment_loss = assignment_cross_entropy_loss(assignment, target, mask, self.options.focal_gamma)
        detection_loss = F.binary_cross_entropy_with_logits(detection, mask.float(), reduction='none')

        return self.options.assignment_loss_scale * assignment_loss, self.options.detection_loss_scale * detection_loss

    def max_over_dims(self, tensor, start_dim=1):
        for dim in range(start_dim, tensor.dim()):
            tensor, _ = torch.max(tensor, dim=dim, keepdim=True)
        return torch.squeeze(tensor)

    def min_over_dims(self, tensor, start_dim=1):
        for dim in range(start_dim, tensor.dim()):
            # Create a mask where tensor is not -inf
            valid_mask = tensor != float('-inf')
            
            # Use the mask to replace -inf with a large positive value, then compute the minimum
            replaced_tensor = torch.where(valid_mask, tensor, torch.tensor(float('inf')).to(tensor.device))
            tensor, _ = torch.min(replaced_tensor, dim=dim, keepdim=True)
            
        return torch.squeeze(tensor)

    def mask_tensor(self, tensor):
        batch_size = tensor.shape[0]
    
        # Compute argmax
        flat_indices = torch.argmax(tensor.view(batch_size, -1), dim=1)
    
        # Convert flat indices to 3D indices
        indices_3d = torch.stack(torch.meshgrid(
            torch.arange(10, device=tensor.device),
            torch.arange(10, device=tensor.device),
            torch.arange(10, device=tensor.device),
            indexing='ij'
        ), dim=-1).reshape(-1, 3)
        a_b_c_indices = indices_3d[flat_indices]  # Shape: (batch_size, 3)
    
        # Create masks for axis 0 and 1
        mask = torch.zeros((batch_size, 10, 10, 10), dtype=bool, device=tensor.device)  # Start with a mask of ones
        for j in range(batch_size):
            # mask[j, a_b_c_indices[j, 1], a_b_c_indices[j, 0], :] = True
            # mask[j, a_b_c_indices[j, 0], a_b_c_indices[j, 1], :] = True
            mask[j, a_b_c_indices[j, 1], :, :] = True
            mask[j, a_b_c_indices[j, 0], :, :] = True
            mask[j, :, a_b_c_indices[j, 1], :] = True
            mask[j, :, a_b_c_indices[j, 0], :] = True
        
        # Apply the mask to the original tensor
        tensor = tensor.masked_fill(mask, float('-inf'))
    
        return tensor, a_b_c_indices
    
    def compute_symmetric_losses(self, assignments: Tuple[torch.Tensor], detections: List[torch.Tensor], targets: Tuple[Tuple[torch.Tensor]]):        
        num_iterations = 2
        symmetric_losses = []
        for iteration in range(num_iterations):
            for permutation in self.event_permutation_tensor.cpu().numpy():
                prepro_losses = []
                if iteration == 0:
                    masks = []
                    for assignment, detection, (target, mask) in zip(assignments, detections, targets[permutation]):
                        assignment_loss, detection_loss = self.particle_symmetric_loss(assignment, detection, target, mask)
                        
                        prepro_losses.append(torch.stack((assignment_loss, detection_loss)))
                else:
                    for assignment, detection, (target, mask), (_, single_mask) in zip(assignments, detections, targets[permutation], targets[np.flip(permutation)]):
                        assignment2, flattened_index = self.mask_tensor(assignment)
                        double_mask = torch.logical_and(mask, single_mask)
                        assignment3 = torch.where(double_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1), assignment2, assignment)
                        assignment_loss, detection_loss = self.particle_symmetric_loss(assignment3, detection, target, double_mask)

                        inf_mask = torch.isinf(assignment_loss)
                        assignment_loss = assignment_loss.masked_fill_(inf_mask, 0)

                        prepro_losses.append(torch.stack((assignment_loss, detection_loss)))

                symmetric_losses.append(torch.stack(prepro_losses))
    
        return torch.stack(symmetric_losses)

    def combine_symmetric_losses(self, symmetric_losses: Tensor) -> Tuple[Tensor, Tensor]:
        # Default option is to find the minimum loss term of the symmetric options.
        # We also store which permutation we used to achieve that minimal loss.
        # combined_loss, _ = symmetric_losses.min(0)
        num_iterations = 2
        symmetric_losses_reshaped = symmetric_losses.view(num_iterations, -1, symmetric_losses.size(-3), symmetric_losses.size(-2), symmetric_losses.size(-1))
        symmetric_losses_reduced = symmetric_losses_reshaped.sum(0) / 2
        total_symmetric_loss = symmetric_losses_reduced.sum((1,2))
        index = total_symmetric_loss.argmin(0)

        combined_loss = torch.gather(symmetric_losses_reduced, 0, index.expand_as(symmetric_losses_reduced))[0]

        # Simple average of all losses as a baseline.
        if self.options.combine_pair_loss.lower() == "mean":
            combined_loss = symmetric_losses_reduced.mean(0)

        # Soft minimum function to smoothly fuse all loss function weighted by their size.
        if self.options.combine_pair_loss.lower() == "softmin":
            weights = F.softmin(total_symmetric_loss, 0)
            weights = weights.unsqueeze(1).unsqueeze(1)
            combined_loss = (weights * symmetric_losses_reduced).sum(0)

        return combined_loss, index

    def symmetric_losses(
        self,
        assignments: List[Tensor],
        detections: List[Tensor],
        targets: Tuple[Tuple[Tensor, Tensor], ...]
    ) -> Tuple[Tensor, Tensor]:
        # We are only going to look at a single prediction points on the distribution for more stable loss calculation
        # We multiply the softmax values by the size of the permutation group to make every target the same
        # regardless of the number of sub-jets in each target particle
        assignments = [prediction + torch.log(torch.scalar_tensor(decoder.num_targets))
                       for prediction, decoder in zip(assignments, self.branch_decoders)]

        # Convert the targets into a numpy array of tensors so we can use fancy indexing from numpy
        targets = numpy_tensor_array(targets)

        # Compute the loss on every valid permutation of the targets
        symmetric_losses = self.compute_symmetric_losses(assignments, detections, targets)

        # Squash the permutation losses into a single value.
        return self.combine_symmetric_losses(symmetric_losses)

    def symmetric_divergence_loss(self, predictions: List[Tensor], masks: Tensor) -> Tensor:
        divergence_loss = []

        for i, j in self.event_info.event_transpositions:
            # Symmetric divergence between these two distributions
            div = jensen_shannon_divergence(predictions[i], predictions[j])

            # ERF term for loss
            loss = torch.exp(-(div ** 2))
            loss = loss.masked_fill(~masks[i], 0.0)
            loss = loss.masked_fill(~masks[j], 0.0)

            divergence_loss.append(loss)

        return torch.stack(divergence_loss).mean(0)
        # return -1 * torch.stack(divergence_loss).sum(0) / len(self.training_dataset.unordered_event_transpositions)

    def add_kl_loss(
            self,
            total_loss: List[Tensor],
            assignments: List[Tensor],
            masks: Tensor,
            weights: Tensor
    ) -> List[Tensor]:
        if len(self.event_info.event_transpositions) == 0:
            return total_loss

        # Compute the symmetric loss between all valid pairs of distributions.
        kl_loss = self.symmetric_divergence_loss(assignments, masks)
        kl_loss = (weights * kl_loss).sum() / masks.sum()

        with torch.no_grad():
            self.log("loss/symmetric_loss", kl_loss, sync_dist=True)
            if torch.isnan(kl_loss):
                raise ValueError("Symmetric KL Loss has diverged.")

        return total_loss + [self.options.kl_loss_scale * kl_loss]

    def add_regression_loss(
            self,
            total_loss: List[Tensor],
            predictions: Dict[str, Tensor],
            targets:  Dict[str, Tensor]
    ) -> List[Tensor]:
        regression_terms = []

        for key in targets:
            current_target_type = self.training_dataset.regression_types[key]
            current_prediction = predictions[key]
            current_target = targets[key]

            current_mean = self.regression_decoder.networks[key].mean
            current_std = self.regression_decoder.networks[key].std

            current_mask = ~torch.isnan(current_target)

            current_loss = regression_loss(current_target_type)(
                current_prediction[current_mask],
                current_target[current_mask],
                current_mean,
                current_std
            )
            current_loss = torch.mean(current_loss)

            with torch.no_grad():
                self.log(f"loss/regression/{key}", current_loss, sync_dist=True)

            regression_terms.append(self.options.regression_loss_scale * current_loss)

        return total_loss + regression_terms

    def add_classification_loss(
            self,
            total_loss: List[Tensor],
            predictions: Dict[str, Tensor],
            targets: Dict[str, Tensor]
    ) -> List[Tensor]:
        classification_terms = []

        for key in targets:
            current_prediction = predictions[key]
            current_target = targets[key]

            weight = None if not self.balance_classifications else self.classification_weights[key]
            current_loss = F.cross_entropy(
                current_prediction,
                current_target,
                ignore_index=-1,
                weight=weight
            )

            classification_terms.append(self.options.classification_loss_scale * current_loss)

            with torch.no_grad():
                self.log(f"loss/classification/{key}", current_loss, sync_dist=True)

        return total_loss + classification_terms

    def training_step(self, batch: Batch, batch_nb: int) -> Dict[str, Tensor]:
        # ===================================================================================================
        # Network Forward Pass
        # ---------------------------------------------------------------------------------------------------
        outputs = self.forward(batch.sources)

        # ===================================================================================================
        # Initial log-likelihood loss for classification task
        # ---------------------------------------------------------------------------------------------------
        symmetric_losses, best_indices = self.symmetric_losses(
            outputs.assignments,
            outputs.detections,
            batch.assignment_targets
        )

        # Construct the newly permuted masks based on the minimal permutation found during NLL loss.
        permutations = self.event_permutation_tensor[best_indices].T
        masks = torch.stack([target.mask for target in batch.assignment_targets])
        masks = torch.gather(masks, 0, permutations)

        # ===================================================================================================
        # Balance the loss based on the distribution of various classes in the dataset.
        # ---------------------------------------------------------------------------------------------------

        # Default unity weight on correct device.
        weights = torch.ones_like(symmetric_losses)

        # Balance based on the particles present - only used in partial event training
        if self.balance_particles:
            class_indices = (masks * self.particle_index_tensor.unsqueeze(1)).sum(0)
            weights *= self.particle_weights_tensor[class_indices]

        # Balance based on the number of jets in this event
        if self.balance_jets:
            weights *= self.jet_weights_tensor[batch.num_vectors]

        # Take the weighted average of the symmetric loss terms.
        masks = masks.unsqueeze(1)
        symmetric_losses = (weights * symmetric_losses).sum(-1) / torch.clamp(masks.sum(-1), 1, None)
        assignment_loss, detection_loss = torch.unbind(symmetric_losses, 1)


        # ===================================================================================================
        # Some basic logging
        # ---------------------------------------------------------------------------------------------------
        with torch.no_grad():
            for name, l in zip(self.training_dataset.assignments, assignment_loss):
                self.log(f"loss/{name}/assignment_loss", l, sync_dist=True)

            for name, l in zip(self.training_dataset.assignments, detection_loss):
                self.log(f"loss/{name}/detection_loss", l, sync_dist=True)

            if torch.isnan(assignment_loss).any():
                raise ValueError("Assignment loss has diverged!")

            if torch.isinf(assignment_loss).any():
                raise ValueError("Assignment targets contain a collision.")

        # ===================================================================================================
        # Start constructing the list of all computed loss terms.
        # ---------------------------------------------------------------------------------------------------
        total_loss = []

        if self.options.assignment_loss_scale > 0:
            total_loss.append(assignment_loss)

        if self.options.detection_loss_scale > 0:
            total_loss.append(detection_loss)

        # ===================================================================================================
        # Auxiliary loss terms which are added to reconstruction loss for alternative targets.
        # ---------------------------------------------------------------------------------------------------
        if self.options.kl_loss_scale > 0:
            total_loss = self.add_kl_loss(total_loss, outputs.assignments, masks, weights)

        if self.options.regression_loss_scale > 0:
            total_loss = self.add_regression_loss(total_loss, outputs.regressions, batch.regression_targets)

        if self.options.classification_loss_scale > 0:
            total_loss = self.add_classification_loss(total_loss, outputs.classifications, batch.classification_targets)

        # ===================================================================================================
        # Combine and return the loss
        # ---------------------------------------------------------------------------------------------------
        total_loss = torch.cat([loss.view(-1) for loss in total_loss])

        self.log("loss/total_loss", total_loss.sum(), sync_dist=True)

        return total_loss.mean()
