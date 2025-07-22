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
        self.options = options
        real_K = self.options.branch_dim * self.options.k - 1

        # -- Classifier Head (whole event) --
        # For each event, flatten all hypothesis/branch/jet/feature into a single vector
        class_input_dim = self.options.branch_dim * self.options.jet_max_dim * self.options.features_dim * real_K
        classifier_layers = []
        class_dims = [class_input_dim] + class_hidden_dims + [real_K]
        for in_d, out_d in zip(class_dims[:-1], class_dims[1:]):
            classifier_layers.append(nn.Linear(in_d, out_d))
            if out_d != real_K:
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

    def forward_scm(self, batch: Batch):
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

        # Mask all logits with true label except for the first one (focus loss on one target only)
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
        masker_k_loss = torch.zeros(events, device=features_arr.device)

        for k in range(K):
            # features_arr[:, k] has shape (events, branches, jets, features)
            # Flatten jets/features for each branch independently; Reshape to (events, branches * jets * features)
            hypo_arr = features_arr[:, k].reshape(events, branches * jets * features)
            logits_k = self.masker(hypo_arr)  # (events, branches)

            masker_k_loss += nn.BCEWithLogitsLoss()(logits_k, pred_truth[:, k].float().to(logits_k.device))

        mask_loss = masker_k_loss.sum()
        
        # # ----------- Top-1 accuracy -----------
        pred_k = torch.argmax(class_logits, dim=1)  # for each hypo, find idx with highest logit; shape: (events,)  
        event_idx = torch.arange(events)            # vector of event‐indices
        
        # for each event e, look up class_truth[e, pred_k[e]]; convert to float -> sum # of correct pred        
        correct_predictions = class_truth[event_idx, pred_k].float().sum()
        top1_acc = correct_predictions / events

        return class_loss, mask_loss, top1_acc

    def training_step(self, batch: Batch, batch_idx: int) -> Dict[str, torch.Tensor]:

        self.on_train_epoch_start()

        class_loss, mask_loss, top1_acc = self.forward_scm(batch)

        total_loss = class_loss + mask_loss

        self.log('train_classifier_loss', class_loss)
        self.log('train_masker_loss', mask_loss)
        self.log('train_total_loss', total_loss)
        self.log('train_top1_acc', top1_acc)

        return total_loss
        
    def validation_step(self, batch: Batch, batch_idx: int) -> Dict[str, torch.Tensor]:

        class_loss, mask_loss, top1_acc = self.forward_scm(batch)
        total_loss = class_loss + mask_loss

        self.log('val_classifier_loss', class_loss, on_epoch=True, prog_bar=True)
        self.log('val_masker_loss', mask_loss, on_epoch=True, prog_bar=True)
        self.log('val_total_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_top1_acc', top1_acc, on_epoch=True, prog_bar=True)

        return {'val_total_loss': total_loss}
    
    def on_train_epoch_start(self):
        for name, module in self.named_children():
            if name not in ['classifier', 'masker']:
                module.eval()
        self.eval()
        self.classifier.train()
        self.masker.train()
        for name, module in self.named_children():
            print(f"{name}: {'train' if module.training else 'eval'}")

    def on_train_batch_start(self, batch, batch_idx):
        for name, module in self.named_children():
            if name not in ['classifier', 'masker']:
                module.eval()
        self.eval()
        self.classifier.train()
        self.masker.train()
        for name, module in self.named_children():
            print(f"{name}: {'train' if module.training else 'eval'}")

