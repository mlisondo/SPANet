import inspect      # for debugging purposes

from typing import Dict, Callable

import numpy as np
import torch

from sklearn import metrics as sk_metrics

from spanet.options import Options
from spanet.dataset.evaluator import SymmetricEvaluator
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork


class JetReconstructionValidation(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetReconstructionValidation, self).__init__(options, torch_script)
        self.evaluator = SymmetricEvaluator(self.training_dataset.event_info)

    @property
    def particle_metrics(self) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        return {
            "accuracy": sk_metrics.accuracy_score,
            "sensitivity": sk_metrics.recall_score,
            "specificity": lambda t, p: sk_metrics.recall_score(~t, ~p),
            "f_score": sk_metrics.f1_score
        }

    @property
    def particle_score_metrics(self) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        return {
            # "roc_auc": sk_metrics.roc_auc_score,
            # "average_precision": sk_metrics.average_precision_score
        }

    def compute_metrics(self, jet_predictions, particle_scores, stacked_targets, stacked_masks):
        event_permutation_group = self.event_permutation_tensor.cpu().numpy()
        num_permutations = len(event_permutation_group)
        num_targets, batch_size = stacked_masks.shape
        particle_predictions = particle_scores >= 0.5

        # PROBE ZONE
        print("Probe Station #1: compute_metrics".center(50, '~'))
        probe(jet_predictions, 'jet_predictions')
        probe(particle_scores, 'particle_scores')
        probe(stacked_targets, 'stacked_targets')
        probe(stacked_masks, 'stacked_masks')
        probe(event_permutation_group, 'event_permutation_group')
        probe(num_permutations, 'num_permutations')
        probe(num_targets, 'num_targets')
        probe(batch_size, 'batch_size')
        probe(particle_predictions, 'particle_predictions')

        # Compute all possible target permutations and take the best performing permutation
        # First compute raw_old accuracy so that we can get an accuracy score for each event
        # This will also act as the method for choosing the best permutation to compare for the other metrics.
        jet_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=bool)
        particle_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=bool)
        for i, permutation in enumerate(event_permutation_group):
            for j, (prediction, target) in enumerate(zip(jet_predictions, stacked_targets[permutation])):
                jet_accuracies[i, j] = np.all(prediction == target, axis=1)

            particle_accuracies[i] = stacked_masks[permutation] == particle_predictions

        jet_accuracies = jet_accuracies.sum(1)
        particle_accuracies = particle_accuracies.sum(1)

        # PROBE ZONE
        print("Probe Station #2: compute_metrics".center(50, '~'))
        probe(particle_accuracies, 'particle_accuracies 1')
        probe(jet_accuracies, 'jet_accuracies 1')

        # Select the primary permutation which we will use for all other metrics.
        chosen_permutations = self.event_permutation_tensor[jet_accuracies.argmax(0)].T
        chosen_permutations = chosen_permutations.cpu()
        permuted_masks = torch.gather(torch.from_numpy(stacked_masks), 0, chosen_permutations).numpy()

        # PROBE ZONE
        print("Probe Station #3: compute_metrics".center(50, '~'))
        probe(chosen_permutations, 'chosen_permutations')
        probe(permuted_masks, 'permuted_masks')

        # Compute final accuracy vectors for output
        num_particles = stacked_masks.sum(0)
        jet_accuracies = jet_accuracies.max(0)
        particle_accuracies = particle_accuracies.max(0)

        # PROBE ZONE
        print("Probe Station #4: compute_metrics".center(50, '~'))
        probe(num_particles, 'num_particles')
        probe(jet_accuracies, 'jet_accuracies 2')
        probe(particle_accuracies, 'particle_accuracies 2')

        # Create the logging dictionaries
        metrics = {f"jet/accuracy_{i}_of_{j}": (jet_accuracies[num_particles == j] >= i).mean()
                   for j in range(1, num_targets + 1)
                   for i in range(1, j + 1)}

        metrics.update({f"particle/accuracy_{i}_of_{j}": (particle_accuracies[num_particles == j] >= i).mean()
                        for j in range(1, num_targets + 1)
                        for i in range(1, j + 1)})

        particle_scores = particle_scores.ravel()
        particle_targets = permuted_masks.ravel()
        particle_predictions = particle_predictions.ravel()

        # PROBE ZONE
        print("Probe Station #5: compute_metrics".center(50, '~'))
        probe(num_particles, 'num_particles')
        probe(jet_accuracies, 'jet_accuracies 2')
        probe(particle_accuracies, 'particle_accuracies 2')

        for name, metric in self.particle_metrics.items():
            metrics[f"particle/{name}"] = metric(particle_targets, particle_predictions)

        for name, metric in self.particle_score_metrics.items():
            metrics[f"particle/{name}"] = metric(particle_targets, particle_scores)

        # Compute the sum accuracy of all complete events to act as our target for
        # early stopping, hyperparameter optimization, learning rate scheduling, etc.
        metrics["validation_accuracy"] = metrics[f"jet/accuracy_{num_targets}_of_{num_targets}"]

        # PROBE ZONE
        print("Probe Station #6: compute_metrics".center(50, '~'))
        probe(metrics, 'metrics')

        return metrics

    def validation_step(self, batch, batch_idx) -> Dict[str, np.float32]:
        # Run the base prediction step
        sources, num_jets, targets, regression_targets, classification_targets = batch
        jet_predictions, particle_scores, regressions, classifications = self.predict(sources)

        batch_size = num_jets.shape[0]
        num_targets = len(targets)

        # PROBE ZONE
        print("Probe Station #1: validation_step".center(50, '~'))
        probe(batch, 'batch')
        probe(sources, 'sources')
        probe(num_jets, 'num_jets')
        probe(targets, 'targets')
        probe(regression_targets, 'regression_targets 1')
        probe(classification_targets, 'classification_targets 1')
        probe(jet_predictions, 'jet_predictions')
        probe(particle_scores, 'particle_scores')
        probe(regressions, 'regressions')
        probe(classifications, 'classifications')
        probe(batch_size, 'batch_size')
        probe(num_targets, 'num_targets')

        # Stack all of the targets into single array, we will also move to numpy for easier the numba computations.
        stacked_targets = np.zeros(num_targets, dtype=object)
        stacked_masks = np.zeros((num_targets, batch_size), dtype=bool)
        for i, (target, mask) in enumerate(targets):
            stacked_targets[i] = target.detach().cpu().numpy()
            stacked_masks[i] = mask.detach().cpu().numpy()

        regression_targets = {
            key: value.detach().cpu().numpy()
            for key, value in regression_targets.items()
        }

        classification_targets = {
            key: value.detach().cpu().numpy()
            for key, value in classification_targets.items()
        }

       # PROBE ZONE
        print("Probe Station #2: validation_step".center(50, '~'))
        probe(stacked_targets, 'stacked_targets')
        probe(stacked_masks, 'stacked_masks')
        probe(regression_targets, 'regression_targets 2')
        probe(classification_targets, 'classification_targets 2')

        metrics = self.evaluator.full_report_string(jet_predictions, stacked_targets, stacked_masks, prefix="Purity/")

        # Apply permutation groups for each target
        for target, prediction, decoder in zip(stacked_targets, jet_predictions, self.branch_decoders):
            for indices in decoder.permutation_indices:
                if len(indices) > 1:
                    prediction[:, indices] = np.sort(prediction[:, indices])
                    target[:, indices] = np.sort(target[:, indices])

       # PROBE ZONE
        print("Probe Station #3: validation_step".center(50, '~'))
        probe(decoder.permutation_indices, 'decoder.permutation_indices')
        probe(prediction, 'prediction')
        probe(target, 'target')

        metrics.update(self.compute_metrics(jet_predictions, particle_scores, stacked_targets, stacked_masks))

        for key in regressions:
            delta = regressions[key] - regression_targets[key]
            
            percent_error = np.abs(delta / regression_targets[key])
            self.log(f"REGRESSION/{key}_percent_error", percent_error.mean(), sync_dist=True)

            absolute_error = np.abs(delta)
            self.log(f"REGRESSION/{key}_absolute_error", absolute_error.mean(), sync_dist=True)

            percent_deviation = delta / regression_targets[key]
            self.logger.experiment.add_histogram(f"REGRESSION/{key}_percent_deviation", percent_deviation, self.global_step)

            absolute_deviation = delta
            self.logger.experiment.add_histogram(f"REGRESSION/{key}_absolute_deviation", absolute_deviation, self.global_step)

        for key in classifications:
            accuracy = (classifications[key] == classification_targets[key])
            self.log(f"CLASSIFICATION/{key}_accuracy", accuracy.mean(), sync_dist=True)

        for name, value in metrics.items():
            if not np.isnan(value):
                self.log(name, value, sync_dist=True)

       # PROBE ZONE
        print("Probe Station #4: validation_step".center(50, '~'))
        probe(metrics, 'metrics')

        return metrics

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

def probe(o, name):
    t = type(o)
    if name != None:
        print(f"\nObject '{name}': Type = {t.__module__}.{t.__name__}")
    else:
        print(f"\nUnnamed Object: Type = {t.__module__}.{t.__name__}")

    if hasattr(o, 'shape'):
        try: print(f"shape: {o.shape}")
        except: pass
    if hasattr(o, 'ndim'):
        try: print(f"ndim: {o.ndim}")
        except: pass
    if hasattr(o, 'dtype'):
        try: print(f"dtype: {o.dtype}")
        except: pass

    if hasattr(o, 'size'):
        try: print(f"size: {o.size}")
        except: pass
    if hasattr(o, 'numel'):
        try: print(f"numel: {o.numel}")
        except: pass
    if hasattr(o, '__len__'):
        try: print(f"__len__: {o.__len__}")
        except: pass

    # try: print(f"attributes: {dir(o)}\n")
    # except Exception:
    #     print(f"attributes: could not be listed\n")
