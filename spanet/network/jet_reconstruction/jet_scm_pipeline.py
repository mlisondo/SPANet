import numpy as np

import torch

from spanet.options import Options
from spanet.dataset.evaluator import SymmetricEvaluator
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork

class JetSecondaryLoader(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetSecondaryLoader, self).__init__(options, torch_script)
        self.evaluator = SymmetricEvaluator(self.training_dataset.event_info)

    @torch.no_grad()
    def topk_data(self, batch):
        sources, _, targets, _, _ = batch
        jet_data, _ = sources[0]  # (events, Njets, F)
        device = jet_data.device
        jet_preds, *_ = self.predict(sources)  # list[len=B]; each (events, K, p_i)
    
        events, Njets, Fdim = jet_data.shape
        B = len(targets) # branches
        K = jet_preds[0].shape[1]
    
        true_idx = [idx_t.to(device) for idx_t, _ in targets] # list[events, p_i]
        true_masks = torch.stack([m.to(device) for _, m in targets]) # (B, ) (branch first)
        partons = torch.tensor([t.shape[1] for t in true_idx],
                               device=device, dtype=torch.long) # (B,)
        max_p = int(partons.max())
    
        pred_truth_list = [] # will hold (events, K) per branch
        feat_list = []  # will hold (events, K, p_i, F) per branch
    
        for b, p_i in enumerate(partons):
            # predicted == truth?
            # jet_preds[b]: (events, K, p_i);  true_idx[b]: (events, p_i)
            matches = (jet_preds[b] == true_idx[b].unsqueeze(1)).all(dim=2)  # (events, K) bool
            pred_truth_list.append(matches)
    
            # gather features
            # reshape to flat list of jet indices, gather, then reshape back
            idx_flat   = jet_preds[b].reshape(events, K * p_i) # (events, K*p_i)
            gathered   = jet_data.gather(1,
                             idx_flat.unsqueeze(-1).expand(-1, -1, Fdim)) # (events, K*p_i, F)
            gathered   = gathered.view(events, K, p_i, Fdim) # (events, K, p_i, F)
    
            # pad along parton dimension so every branch has length max_p
            if p_i < max_p:
                gathered = F.pad(gathered, (0, 0, # features dim
                                            0, max_p-p_i))# pad p_i→max_p
            feat_list.append(gathered)
    
        # stack into final tensors
        pred_truth   = torch.stack(pred_truth_list, dim=2) # (events, K, B)
        features_arr = torch.stack(feat_list,     dim=2) # (events, K, B, max_p, F)
    
        # True => every branch's prediction matches
        # its mask, and at least one branch is true
        mask_matrix  = true_masks.permute(1, 0) # (events, B)
        class_truth  = (pred_truth == mask_matrix.unsqueeze(1)).all(dim=2) # (events, K)
        class_truth &= mask_matrix.any(dim=1, keepdim=True) # require >=1 True mask
    
        return pred_truth, true_masks, features_arr, class_truth




#         # ------------------ experimental for more rigorous masker ------------------
#         targ_arr     = np.zeros((branches, events, max_p), dtype=int)                   # The (possibly padded/filled) ground-truth assignments
#         for event in range(eventsvents):
#             # 1. Build set of available jets for this event based on mask (indices 0..max_jet-1 where mask is True)
#             avail = set(np.flatnonzero(jet_mask[event]))

#             # 2. Loop over all branches: assign pre-specified jets, track locations of -1s (unassigned jets)
#             neg1_locs = []  # Will hold tuples (branch, idx) for all -1s across all branches for this event
#             for branch in range(branches):
#                 tarr = true_idx[branch][event]  # True jet indices for this branch/event (may contain -1)
#                 targ_arr[branch, event, :] = tarr  # Copy to working array so we can modify in-place
#                 for idx, jet in enumerate(tarr):
#                     if jet >= 0:
#                         # Remove any pre-assigned jet from available set to ensure unique usage across branches
#                         avail.discard(jet)
#                     else:
#                         # Save position of -1 for later joint filling
#                         neg1_locs.append((branch, idx))

#             # 3. Randomly assign unique available jets to all -1s across all branches for this event
#             shuffled = np.array(list(avail))
#             np.random.shuffle(shuffled)
#             for (branch, idx), jet in zip(neg1_locs, shuffled):
#                 targ_arr[branch, event, idx] = jet  # Fill -1 position with a unique available jet

#             # 4. Evaluate predicted assignments and fill features for each hypothesis (k) per branch
#             for branch in range(branches):
#                 targ = targ_arr[branch, event, :]  # Final unique assignment for this branch/event
#                 for k in range(K):
#                     pred = jet_preds[branch][event, k, :]  # Model-predicted jet indices for hypothesis k
#                     # Compare prediction to truth (after -1 filling): True if exact match, else False
#                     pred_truth[event, k, branch] = np.array_equal(pred, targ)
#                     # Populate jet features for this hypothesis/branch (always, regardless of match)
#                     for j, jet_idx in enumerate(pred):
#                         features_arr[event, k, branch, j, :] = jet_data[event, jet_idx, :]
#                 # If this branch/event is not reconstructable (mask is False), force pred_truth to all False
#                 if not true_masks[branch, event]:
#                     pred_truth[event, :, branch] = False

#             # 5. Evaluate if the full event-level assignment matches the mask for all branches (top-level metric)
#             for k in range(K):
#                 # Returns True if ALL branch assignments match their respective masks AND at least one is reconstructable
#                 class_truth[event, k] = (
#                     np.all(pred_truth[event, k, :] == true_masks[:, event]) and
#                     np.any(true_masks[:, event])
#                 )
#         return (
#     torch.from_numpy(pred_truth).to(device),
#     torch.from_numpy(targ_arr).to(device),
#     torch.from_numpy(true_masks).to(device),
#     torch.from_numpy(features_arr).to(device),
#     torch.from_numpy(class_truth).to(device)
# )
