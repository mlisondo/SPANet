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

        self.w_prior = options.prior_weight_end
        self.w_inc = 1.0 - options.prior_weight_end

    @torch.no_grad()
    def evaluate_scm_batch(self, batch: Batch) -> Dict[str, np.ndarray]:
        pred_truth, canon_masks, features_arr, class_truth, canon_idx, jet_preds_tensor, jet_mult = self.topk_data(batch)
        cand_keep = self.candidate_mute_mask(jet_preds_tensor)
        cand_kpm = ~cand_keep
        E, K, B, J, F = features_arr.shape

        inclusive_ft = features_arr.clone()
        prior_ft = features_arr[..., :3].contiguous()

        # ================== masker ==================
        (inclusive_bt, inclusive_ct, inclusive_mask_logits, 
        prior_bt, prior_ct, prior_mask_logits) = self.masker(
            inclusive_X = inclusive_ft, prior_X = prior_ft
        )

        # ~~~~~ total ~~~~~
        mask_logits = (self.w_prior * prior_mask_logits) + (self.w_inc * inclusive_mask_logits)     # NOTE: THIS IS A WEIGHTED GUESS, LOSS FUNCTION IS NOT AS BIASED.
        mask_probs = torch.sigmoid(mask_logits)
        mask_preds = (mask_probs > 0.5).long()

        # ~~~~~ inclusive-only ~~~~~
        inc_mask_probs = torch.sigmoid(inclusive_mask_logits)
        inc_mask_preds = (inc_mask_probs > 0.5).long()

        # ~~~~~ prior-only ~~~~~
        p_mask_probs   = torch.sigmoid(prior_mask_logits)
        p_mask_preds   = (p_mask_probs > 0.5).long()

        # ================== classifier ==================
        (inclusive_logits, inclusive_ct, global_inclusive,
        prior_logits, prior_ct, global_prior) = self.classifier(
            inclusive_bt = inclusive_bt, prior_bt = prior_bt,
            inclusive_ct = inclusive_ct, prior_ct = prior_ct,
            candidate_kpm_inclusive = cand_kpm, candidate_kpm_prior = cand_kpm
        )
        neg_inf = torch.tensor(float("-inf"), device=inclusive_logits.device, dtype=inclusive_logits.dtype)

        # ~~~~~ total ~~~~~
        class_logits = (self.w_prior * prior_logits) + (self.w_inc * inclusive_logits)     # NOTE: THIS IS A WEIGHTED GUESS, LOSS FUNCTION IS NOT BIASED.
        masked_logits = class_logits.masked_fill(~cand_keep, neg_inf) # has to be "NOT"edm, should be applied to invalid entries, not valid ones
        class_probs = torch.softmax(masked_logits, dim=1)  # sums to 1 over valid K
        class_preds = class_probs.argmax(dim=1)

        # ~~~~~ inclusive-only ~~~~~
        inc_masked_logits = inclusive_logits.masked_fill(~cand_keep, neg_inf)
        inc_class_probs = torch.softmax(inc_masked_logits, dim=1)
        inc_class_preds = inc_class_probs.argmax(dim=1)

        # ~~~~~ prior-only ~~~~~
        p_masked_logits = prior_logits.masked_fill(~cand_keep, neg_inf)
        p_class_probs = torch.softmax(p_masked_logits, dim=1)
        p_class_preds = p_class_probs.argmax(dim=1)

        raw_valid = cand_keep.any(dim=-1)  # (E,)
        true_masks = canon_masks.T.contiguous()

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
            "cand_keep":   cand_keep,
            "inclusive_class_logits" : inclusive_logits,    # ICL
            "inclusive_class_probs" : inc_class_probs,      # ICP    
            "inclusive_class_preds" : inc_class_preds,      # ICPd
            "prior_class_logits" : prior_logits,    # PCL
            "prior_class_probs" : p_class_probs,    # PCP
            "prior_class_preds" : p_class_preds,    # PCPd
            "inclusive_mask_logits" : inclusive_mask_logits,    # IML
            "inclusive_mask_probs" : inc_mask_probs,            # IMP
            "inclusive_mask_preds" : inc_mask_preds,            # IMPd
            "prior_mask_logits" : prior_mask_logits,    # PML
            "prior_mask_probs" : p_mask_probs,          # PMP
            "prior_mask_preds" : p_mask_preds           # PMPd
        }