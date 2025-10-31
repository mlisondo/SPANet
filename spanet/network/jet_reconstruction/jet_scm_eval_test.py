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
        pred_truth, canon_masks, features_arr, class_truth, canon_idx, jet_preds_tensor, jet_mult = self.topk_data(batch)
        E, K, B, J, F = features_arr.shape

        valid_mask = self.candidate_mute_mask(jet_preds_tensor) # this returns a keep/valid mask, True -> candidate is valid and should be used. must be "NOT"ed if used as attn_mask
        branch_kpm = (~valid_mask).unsqueeze(-1).expand(-1, -1, B)  # (E, K, B)
        branch_kpm = branch_kpm.reshape(-1, B).contiguous()       # (E*K, B)

        inclusive_ft = features_arr.clone()
        prior_ft = features_arr[..., :3].contiguous()

        # ================== masker ==================
        (inclusive_bt, inclusive_ct, inclusive_mask_logits, 
        prior_bt, prior_ct, prior_mask_logits) = self.masker(
            inclusive_X = inclusive_ft, prior_X = prior_ft,
            inclusive_branch_kpm = branch_kpm, prior_branch_kpm = branch_kpm
        )

        mask_logits = prior_mask_logits * 2 + inclusive_mask_logits     # NOTE: THIS IS A WEIGHTED GUESS, LOSS FUNCTION IS NOT AS BIASED.
        mask_probs = torch.sigmoid(mask_logits)
        mask_preds = (mask_probs > 0.5).long()

        # ================== classifier ==================
        (inclusive_logits, inclusive_ct, global_inclusive,
        prior_logits, prior_ct, global_prior) = self.classifier(
            inclusive_bt = inclusive_bt, prior_bt = prior_bt,
            inclusive_ct = inclusive_ct, prior_ct = prior_ct,
            branch_kpm_inclusive = branch_kpm, branch_kpm_prior = branch_kpm
        )

        class_logits = prior_logits * 2 + inclusive_logits     # NOTE: THIS IS A WEIGHTED GUESS, LOSS FUNCTION IS NOT BIASED.
        neg_inf = torch.tensor(float("-inf"), device=inclusive_logits.device, dtype=inclusive_logits.dtype)
        masked_logits = class_logits.masked_fill(~valid_mask, neg_inf) # has to be "NOT"edm, should be applied to invalid entries, not valid ones
        class_probs = torch.softmax(masked_logits, dim=1)  # sums to 1 over valid K
        class_preds = class_probs.argmax(dim=1)

        raw_valid = valid_mask.any(dim=-1)  # (E,)

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
            "true_masks":   canon_masks,    # TM
            "raw_valid":    raw_valid,      # RV
            "jet_mult":     jet_mult,       # JM
            "valid_mask":   valid_mask
        }