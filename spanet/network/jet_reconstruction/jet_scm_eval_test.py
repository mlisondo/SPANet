import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict
from spanet.options import Options
from spanet.network.jet_reconstruction.jet_scm_training_validation import SCM_Training_Val
from spanet.dataset.types import Batch

class SCM_Eval_Test(SCM_Training_Val):
    def __init__(self, options: Options, torch_script: bool = False):
        super().__init__(options, torch_script)

    @torch.no_grad()
    def evaluate_scm_batch(self, batch: Batch) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary with:
        CLASSIFIER:
        - CL  -> class_logits: raw logits (events, K)
        - CP  -> class_probs: softmax probabilities (events, K)
        - CPd -> class_preds: index per event (events,) ; if multiple positives, pick the best-scoring positive.
        MASKER:
        - ML  -> mask_logits: raw logits (events, K, branches)
        - MP  -> mask_probs: sigmoid probabilities (events, K, branches)
        - MPd -> mask_preds: predicted mask (events, K, branches)
        FROM PIPELINE:
        - CT -> class_truth: (events, K)  # multi-hot allowed
        - MT -> mask_truth: (events, K, branches)
        - FA -> features_arr: (events, K, branches, jets, features)
        - TM -> true_masks: (events, branches)
        """
        pred_truth, true_masks, features_arr, class_truth, true_idx, jet_preds_tensor = self.topk_data(batch)
        true_masks = true_masks.permute(1, 0)  # (events, branches)
        events, K, branches, jets, features = features_arr.shape
    
        # ----------- Classifier Head -----------
        # class_in = features_arr.reshape(events, -1) # this was for the MLP
        # class_logits = self.classifier(class_in)                 # (events, K)
        class_logits, token_scores = self.classifier(features_arr)  # token_scores are not used
        class_probs  = torch.softmax(class_logits, dim=1)        # (events, K)
    
        # LSE-style evaluation: if any positives exist for an event, choose the
        # highest-logit class among the positive set; otherwise fall back to global argmax.
        pos = class_truth.bool()                                 # (events, K)
        has_pos = pos.any(dim=1)                                 # (events,)
        neg_inf = torch.finfo(class_logits.dtype).min
        pos_only = torch.where(pos, class_logits, torch.full_like(class_logits, neg_inf))
        pos_choice = pos_only.argmax(dim=1)                      # (events,)
        global_choice = class_logits.argmax(dim=1)               # (events,)
        class_preds = torch.where(has_pos, pos_choice, global_choice)
    
        # ----------- Masker Head -----------
        mask_logits_list, mask_probs_list, mask_preds_list = [], [], []
        for k in range(K):
            # hypo_arr = features_arr[:, k].reshape(events, branches * jets * features) # this was for the MLP
            hypo_arr = features_arr[:, k]
            logits_k = self.masker(hypo_arr)                     # (events, branches)
            probs_k  = torch.sigmoid(logits_k)                   # (events, branches)
            preds_k  = (probs_k > 0.5).long()                    # (events, branches)
    
            mask_logits_list.append(logits_k.unsqueeze(1))
            mask_probs_list.append(probs_k.unsqueeze(1))
            mask_preds_list.append(preds_k.unsqueeze(1))
    
        mask_logits = torch.cat(mask_logits_list, dim=1)         # (events, K, branches)
        mask_probs  = torch.cat(mask_probs_list,  dim=1)
        mask_preds  = torch.cat(mask_preds_list,  dim=1)
    
        # ----------- Return (same keys / order) -----------
        return {
            "class_logits": class_logits,        # CL
            "class_probs":  class_probs,         # CP
            "class_preds":  class_preds,         # CPd (best positive if available)
            "mask_logits":  mask_logits,         # ML
            "mask_probs":   mask_probs,          # MP
            "mask_preds":   mask_preds,          # MPd
            "class_truth":  class_truth,         # CT
            "mask_truth":   pred_truth,          # MT
            "features_arr": features_arr,        # FA
            "true_masks":   true_masks           # TM
        }
