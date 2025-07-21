import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List, Dict
from spanet.options import Options
from spanet.network.jet_reconstruction.jet_scm_pipeline import JetSecondaryLoader
from spanet.dataset.types import Batch

class SCM_Training_Val(JetSecondaryLoader):
    def __init__(self, options: Options, class_hidden_dims: List[int], mask_hidden_dims: List[int] = None, torch_script: bool = False):
        '''
        input_dim and hidden_dims are defined in train_scm.py

        class_input_dim default = 30
        mask_input_dim defual = 30

        class_hidden_dims deafult = [30,64]
        mask_hidden_dims default = [30, 32]
        '''
        super(SCM_Training_Val, self).__init__(options, torch_script)
        self.options = Options

        # -- Classifier Head (whole event) --
        # For each event, flatten all hypothesis/branch/jet/feature into a single vector
        class_input_dim = self.options.branch_dim * self.options.jet_max_dim * self.options.features_dim * self.options.k
        classifier_layers = []
        class_dims = [class_input_dim] + class_hidden_dims + [self.options.k]
        for in_d, out_d in zip(class_dims[:-1], class_dims[1:]):
            classifier_layers.append(nn.Linear(in_d, out_d))
            if out_d != self.options.k:
                classifier_layers.append(nn.ReLU())
        self.classifier = nn.Sequential(*classifier_layers)

        # -- Masker Head (branch-level) --
        # Masker processes each branch in each hypothesis individually
        masker_input_dim = self.options.branch_dim * self.options.jet_max_dim * self.options.features_dim
        masker_layers = []
        mask_dims = [masker_input_dim] + mask_hidden_dims + [self.options.branch_dim]
        for in_d, out_d in zip(mask_dims[:-1], mask_dims[1:]):
            masker_layers.append(nn.Linear(in_d, out_d))
            if out_d != self.options.branch_dim:
                masker_layers.append(nn.ReLU())
        self.masker = nn.Sequential(*masker_layers)

    def forward(self, batch: Batch):
        pred_truth, true_masks, features_arr, class_truth = self.topk_data(batch)
        true_masks = true_masks.permute(1, 0) # true_masks: (branches, events) -> (events, branches) 
        events, K, branches, jets, features = features_arr.shape

        # ----------- Classifier -----------
        # Prepare classifier input: flatten event features for MLP
        # features_arr: (events, K, branches, jets, features)
        class_in = features_arr.reshape(events, -1)  # (events, K * branches * jets * features)

        # Forward pass through classifier head
        class_logits = self.classifier(class_in)  # (events, K)

        # Prepare ground truth: integer format and get index of first positive
        class_truth_int = class_truth.to(torch.int)  # (events, K)
        class_first = torch.argmax(class_truth_int, dim=1)  # (events,)
        has_truth = torch.any(class_truth_int == 1, dim=1)  # (events,)

        mask = class_truth.bool().clone() # which positions are "correct"
        has_true = mask.any(dim=1) # rows that actually have a True
        rows = torch.arange(events, device=class_logits.device)
        
        # Clear the mask at the position we want to keep
        mask[rows[has_true], class_first[has_true]] = False
        
        # Now mask contains True exactly where we want to set -inf
        neg_inf = torch.finfo(class_logits.dtype).min # safer than -inf for some ops
        masked_logits = class_logits.masked_fill(mask, neg_inf)

        # Cross-entropy loss, summed only over events with at least one true label
        class_loss = nn.CrossEntropyLoss(reduction="none")(class_logits, class_first)[has_truth].sum()


        # ----------- Masker -----------
        # For each hypothesis k, process branches independently
        masker_k_loss = torch.zeros(events)
        for k in range(K):
            # features_arr[:, k] has shape (events, branches, jets, features)
            # Flatten jets/features for each branch independently; Reshape to (events, branches * jets * features)
            hypo_arr = features_arr[:, k].reshape(events, branches * jets * features)
            logits_k = self.masker(hypo_arr)  # (events, branches)

            masker_k_loss  += nn.BCEWithLogitsLoss()(logits_k, pred_truth[:,k])

        mask_loss = masker_k_loss.sum()
        

        # # ----------- Top-1 accuracy -----------
        # # Select, for each event, the hypothesis with the highest predicted score
        # pred_k = class_logits.argmax(dim=1)  # (events,)
        # # For each event, check if the predicted hypothesis is correct
        # # class_truth: (events, K), class_truth[batch_index, pred_k] gives whether prediction is correct
        # top1_acc = (class_truth[torch.arange(class_truth.size(0)), pred_k] > 0).float().mean()

        return class_loss, mask_loss#, top1_acc
    
    def training_step(self, batch: Batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        # class_loss, mask_loss, top1_acc = self.forward(batch)
        class_loss, mask_loss = self.forward(batch)

        total_loss = class_loss + mask_loss

        self.log('train_classifier_loss', class_loss)
        self.log('train_masker_loss', mask_loss)
        self.log('train_total_loss', total_loss)
        # self.log('train_top1_acc', top1_acc)

        return total_loss
    
    def validation_step(self, batch: Batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        # class_loss, mask_loss, top1_acc = self.forward(batch)
        class_loss, mask_loss = self.forward(batch)

        total_loss = class_loss + mask_loss

        self.log('train_classifier_loss', class_loss)
        self.log('train_masker_loss', mask_loss)
        self.log('train_total_loss', total_loss)
        # self.log('train_top1_acc', top1_acc)

        # return {'val_loss': total_loss, 'val_top1_acc': top1_acc}
        return {'val_loss': total_loss}
