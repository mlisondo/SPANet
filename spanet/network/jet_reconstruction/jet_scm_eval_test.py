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
        self.eval()
        pred_truth, true_masks, features_arr, class_truth, true_idx, jet_preds_tensor, jet_mult = self.topk_data(batch)
        true_masks = true_masks.permute(1, 0)  # (E, B)
        E, K, B, J, F = features_arr.shape

        # deduplicate K-candidates via jet index patterns
        valid_mask = self._dedup_valid_mask(jet_preds_tensor)  # (E, K), True = keep

        # classifier with masking
        class_logits, token_scores, out_valid_mask = self.classifier(features_arr, valid_mask)
        class_probs = class_logits.exp()            # sums to 1 over valid K
        class_preds = class_probs.argmax(dim=1)

        # masker head (optional: suppress duplicates for clarity)
        mask_logits_list, mask_probs_list, mask_preds_list = [], [], []
        for k in range(K):
            logits_k = self.masker(features_arr[:, k])                 # (E, B)
            if out_valid_mask is not None:
                vmk = out_valid_mask[:, k].unsqueeze(1)               # (E,1)
                logits_k = logits_k.masked_fill(~vmk, neg_inf)
            probs_k  = torch.sigmoid(logits_k)
            preds_k  = (probs_k > 0.5).long()
            mask_logits_list.append(logits_k.unsqueeze(1))
            mask_probs_list.append(probs_k.unsqueeze(1))
            mask_preds_list.append(preds_k.unsqueeze(1))

        mask_logits = torch.cat(mask_logits_list, dim=1)              # (E, K, B)
        mask_probs  = torch.cat(mask_probs_list,  dim=1)
        mask_preds  = torch.cat(mask_preds_list,  dim=1)

        raw_valid = true_masks.any(dim=-1)  # (E,)

        return {
            "class_logits": class_logits,   # CL
            "class_probs":  class_probs,    # CP (zeros on duplicates)
            "class_preds":  class_preds,    # CPd
            "mask_logits":  mask_logits,    # ML
            "mask_probs":   mask_probs,     # MP
            "mask_preds":   mask_preds,     # MPd
            "class_truth":  class_truth,    # CT
            "pred_truth":   pred_truth,     # PT
            "features_arr": features_arr,   # FA
            "true_masks":   true_masks,     # TM
            "raw_valid":    raw_valid,      # RV
            "jet_mult":     jet_mult,       # JM
            "valid_mask":   out_valid_mask  # for monitoring
        }
