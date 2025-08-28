import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict
from spanet.options import Options
from spanet.network.jet_reconstruction.jet_scm_training_validation import SCM_Training_Val
from spanet.dataset.types import Batch

def fuse_duplicate_k(features_arr, class_logits, decimals=6):
    # features_arr: [E, K, B, J, F]
    # class_logits: [E, K]
    E, K, *_ = features_arr.shape
    device = features_arr.device
    dtype = class_logits.dtype
    neg_inf = torch.finfo(dtype).min

    scale = 10.0 ** decimals
    flat = features_arr.reshape(E, K, -1)
    q = torch.round(flat * scale).to(torch.int64)

    q2 = q.view(E * K, -1)
    ev = torch.arange(E, device=device).repeat_interleave(K).unsqueeze(1)
    keys = torch.cat([ev.to(torch.int64), q2], dim=1)
    uniq, inverse = torch.unique(keys, dim=0, return_inverse=True)  # inverse: [E*K]
    G = uniq.size(0)

    logits = class_logits.reshape(-1)
    m = torch.full((G,), neg_inf, dtype=dtype, device=device)
    m.scatter_reduce_(0, inverse, logits, reduce="amax")
    s = torch.zeros(G, dtype=dtype, device=device)
    s.scatter_add_(0, inverse, torch.exp(logits - m[inverse]))
    agg = m + torch.log(s.clamp_min(1e-20))

    k_idx = torch.arange(K, device=device).repeat(E)          # [E*K]
    rep_k = torch.full((G,), K, device=device, dtype=torch.int64)
    rep_k.scatter_reduce_(0, inverse, k_idx, reduce="amin")

    group_event = uniq[:, 0].to(torch.int64)                  # [G]
    counts = torch.bincount(group_event, minlength=E)         # [E]
    max_G = int(counts.max().item())
    starts = torch.zeros(E + 1, dtype=torch.int64, device=device)
    starts[1:] = counts.cumsum(0)

    agg_padded = torch.full((E, max_G), neg_inf, dtype=dtype, device=device)
    start = 0
    for e in range(E):
        g = int(counts[e])
        if g:
            agg_padded[e, :g] = agg[start:start+g]
            start += g

    return agg_padded, starts, rep_k, counts

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
        pred_truth, true_masks, features_arr, class_truth, true_idx, jet_preds_tensor, jet_mult = self.topk_data(batch)
        true_masks = true_masks.permute(1, 0)  # (events, branches)
        events, K, branches, jets, features = features_arr.shape
    
        # ----------- Classifier Head -----------
        # class_in = features_arr.reshape(events, -1) # this was for the MLP
        # class_logits = self.classifier(class_in)                 # (events, K)
        class_logits, token_scores = self.classifier(features_arr)  # token_scores are not used
        class_probs  = torch.softmax(class_logits, dim=1)        # (events, K)
    
        agg_logits, starts, rep_k, counts = fuse_duplicate_k(features_arr, class_logits, decimals=6, mode="logsumexp")
        # Softmax over unique sets (padding is -inf so safe)
        group_probs = torch.softmax(agg_logits, dim=1) # [E, max_groups]
        g_pred = agg_logits.argmax(dim=1) # [E]
        global_gid = starts[:-1] + g_pred # [E]
        class_preds = rep_k[global_gid] # [E]
    
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

        # require at least one branch to be reconstructable 
        raw_valid = true_masks.any(dim=-1) # (E,)
    
        # ----------- Return (same keys / order) -----------
        return {
            "class_logits": class_logits,        # CL
            "class_probs":  class_probs,         # CP
            "class_preds":  class_preds,         # CPd (best positive if available)
            "mask_logits":  mask_logits,         # ML
            "mask_probs":   mask_probs,          # MP
            "mask_preds":   mask_preds,          # MPd
            "class_truth":  class_truth,         # CT
            "pred_truth":   pred_truth,          # PT
            "features_arr": features_arr,        # FA
            "true_masks":   true_masks,          # TM
            "raw_valid":    raw_valid,           # RV
            "jet_mult": jet_mult                 # JM
        }
