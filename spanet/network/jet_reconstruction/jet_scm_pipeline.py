import numpy as np

import torch

from spanet.options import Options
from spanet.dataset.evaluator import SymmetricEvaluator
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork

class JetSecondaryLoader(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetSecondaryLoader, self).__init__(options, torch_script)
        self.evaluator = SymmetricEvaluator(self.training_dataset.event_info)

    def topk_data(self, batch):
        sources, _, targets, _, _ = batch
        device = sources[0][0].device

        with torch.no_grad():
            jet_preds, _, _, _ = self.predict(sources)  # shape [branches]: (events, jet_idx, K) = [2](32, 3, K)

        jet_data, _ = sources[0]

        events, _, features = jet_data.shape
        branches = len(targets)
        K = jet_preds[0].shape[1]

        true_idx   = [None] * branches
        true_masks = torch.zeros((branches, events), dtype=torch.bool, device=device)
        partons    = torch.zeros((branches,), dtype=torch.long, device=device)

        for i, (idx_t, mask_t) in enumerate(targets):               # each targets[i].indices is (events, p_i), mask is (events,)
            true_idx[i]   = idx_t
            partons[i]    = idx_t.shape[1]                  # p_i (for ttbar = 3)
            true_masks[i] = mask_t

        max_p = torch.max(partons).item()

        pred_truth   = torch.zeros((events, K, branches), dtype=torch.bool, device=device)                      # Whether each hypothesis matches the truth
        class_truth  = torch.zeros((events, K), dtype=torch.bool, device=device)                                # Whether the whole K'th predicted event is true
        features_arr = torch.zeros((events, K, branches, max_p, features), dtype=torch.float, device=device)    # The per-hypothesis jet features for every event

        for event in range(events):
            for branch in range(branches):
                targ = true_idx[branch][event, :].long()
                for k in range(K):
                    pred = jet_preds[branch][event, k, :].long()
                    if torch.equal(pred, targ):
                        pred_truth[event, k, branch] = True
                    for j, jet_idx in enumerate(pred):
                        features_arr[event, k, branch, j, :] = jet_data[event, jet_idx, :]
            for k in range(K):
                # class_truth[event, k]: True if all pred_truth[event, k, :] matches true_masks[:, event], and at least one mask is True
                if torch.all(pred_truth[event, k, :] == true_masks[:, event]) and torch.any(true_masks[:, event]):
                    class_truth[event, k] = True

        return pred_truth, true_masks, features_arr, class_truth




#         # ------------------ experimental for more rigorous masker ------------------
#         targ_arr     = np.zeros((branches, events, max_p), dtype=int)                   # The (possibly padded/filled) ground-truth assignments
#         for event in range(events):
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
