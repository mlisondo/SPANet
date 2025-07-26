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
        - CPd -> class_preds: index of max logit per event (events,)

        MASKER:
        - ML  -> mask_logits: raw logits (events, K, branches)
        - MP  -> mask_probs: sigmoid probabilities (events, K, branches)
        - MPd -> mask_preds: predicted mask (events, K, branches)

        FROM PIPELINE:
        - CT -> class_truth: ground truth for classifier (events, K)
        - MT -> mask_truth: ground truth for masker (events, K, branches)
        - FA -> features_arr: all jet features (events, K, branches, jets, features)
        - TM -> true_masks: branch-level masks (events, branches)

        i.e. EVERYTHING !
        """

        pred_truth, true_masks, features_arr, class_truth, true_idx, jet_preds_tensor = self.topk_data(batch)
        true_masks = true_masks.permute(1, 0)  # (events, branches)
        events, K, branches, jets, features = features_arr.shape

        # ----------- Classifier Head (No masking or loss) -----------
        class_in = features_arr.reshape(events, -1)
        class_logits = self.classifier(class_in)
        class_probs = torch.softmax(class_logits, dim=1)  # (events, K)
        class_preds = torch.argmax(class_logits, dim=1)   # (events,)

        # ----------- Masker Head (No masking or loss) -----------
        mask_logits_list = []
        mask_probs_list = []
        mask_preds_list = []
        for k in range(K):
            hypo_arr = features_arr[:, k].reshape(events, branches * jets * features)
            logits_k = self.masker(hypo_arr)  # (events, branches)
            probs_k = torch.sigmoid(logits_k) # (events, branches)
            preds_k = (probs_k > 0.5).long()  # (events, branches), binary mask prediction

            # Add K axis for stacking
            mask_logits_list.append(logits_k.unsqueeze(1))
            mask_probs_list.append(probs_k.unsqueeze(1))
            mask_preds_list.append(preds_k.unsqueeze(1))

        # Stack along K axis
        mask_logits = torch.cat(mask_logits_list, dim=1)  # (events, K, branches)
        mask_probs = torch.cat(mask_probs_list, dim=1)
        mask_preds = torch.cat(mask_preds_list, dim=1)

        # ----------- Dual Head (No masking or loss) -----------

        return {
            "class_logits": class_logits,        # (events, K)                  CL
            "class_probs": class_probs,          # (events, K)                  CP
            "class_preds": class_preds,          # (events,)                    CPd
            "mask_logits": mask_logits,          # (events, K, branches)        ML
            "mask_probs": mask_probs,            # (events, K, branches)        MP
            "mask_preds": mask_preds,            # (events, K, branches)        MPd
            "class_truth": class_truth,          # (events, K)                  CT
            "mask_truth": pred_truth,            # (events, K, branches)        MT
            "features_arr": features_arr,        # (events, K, branches, jets, features); features = 5: (btag, eta mass, phi, pt)
            "true_masks": true_masks             # (events, branches)
        }
