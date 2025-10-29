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

        valid_mask = self._dedup_valid_mask(jet_preds_tensor)
        branch_kpm = (~cand_kpm).unsqueeze(-1).expand(-1, -1, B)  # (E, K, B)
        branch_kpm = branch_kpm.reshape(-1, B).contiguous()       # (E*K, B)

        inclusive_ft = features_arr.clone()
        prior_ft = features_arr[..., :3].contiguous()

        # ================== masker ==================
        # masker will be avoided for now
        (inclusive_bt, inclusive_ct, inclusive_mask_logits, 
        prior_bt, prior_ct, prior_mask_logits) = self.masker(
            inclusive_X = inclusive_ft, prior_X = prior_ft,
            inclusive_branch_kpm = branch_kpm, prior_branch_kpm = branch_kpm
        )

        # ================== classifier ==================
        (inclusive_logits, inclusive_ct, global_inclusive,
        prior_logits, prior_ct, global_prior) = self.classifier(
            inclusive_bt = inclusive_bt, prior_bt = prior_bt,
            inclusive_ct = inclusive_ct, prior_ct = prior_ct,
            branch_kpm_inclusive = branch_kpm, branch_kpm_prior = branch_kpm
        )

        neg_inf = torch.tensor(float("-inf"), device = inclusive_logits, dtype = inclusive_logits)