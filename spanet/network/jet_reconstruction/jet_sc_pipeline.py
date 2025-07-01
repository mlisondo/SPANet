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
        jet_preds, _, _, _ = self.predict(sources)  # shape [branches]: (events, jet_idx, K) = [2](32, 3, K)

        jet_data_tensor, jet_masks_tensor = sources[0]
        jet_data = jet_data_tensor.detach().cpu().numpy()   # shape: (32, 10, 5)
        jet_mask = jet_masks_tensor.detach().cpu().numpy()  # shape: (32, 10)

        events, max_jet, features = jet_data.shape
        branches = len(targets)
        K = jet_preds[0].shape[2]

        true_idx   = [None] * branches
        true_masks = np.zeros((branches, events), dtype = bool)
        partons    = np.zeros((branches), dtype = int)

        for i, (idx_t, mask_t) in enumerate(targets):       # each targets[i].indices is (events, p_i), mask is (events,)
            idx_np        = idx_t.detach().cpu().numpy()    # (events, p_i)
            true_idx[i]   = idx_np.copy()
            partons[i]    = idx_np.shape[1]                 # p_i (for ttbar = 3)
            true_masks[i] = mask_t.detach().cpu().numpy()

        max_p = np.max(partons)

        for i, decoder in enumerate(self.branch_decoders):  # apply permutation symmetries to ground truth and predictions
            for grp in decoder.permutation_indices:
                if len(grp) > 1:
                    true_idx[i][:, grp]     = np.sort(true_idx[i][:, grp], axis=1)
                    jet_preds[i][:, grp, :] = np.sort(jet_preds[i][:, grp, :], axis=1)

        # preallocate outputs
        truth = np.zeros((events, K, branches), dtype=bool)
        features_arr = np.zeros((events, K, branches, max_p, features), dtype=float)
        targ_arr = np.zeros((branches, events, max_p), dtype=int)

        for event in range(events):
            available_jets_mask = jet_mask[event].copy()
            for branch in range(branches):
                targ = np.array(true_idx[branch][event, :], dtype = int) # needs to be an array to be iterable (i think)
                for k in range(K):
                    pred = jet_preds[branch][event, :, k]
                    if np.array_equal(pred, targ):  # each shape: jet_idx
                        truth[event, k, branch] = True
                    if not true_masks[branch, event]:
                        used_jet_idx = np.ones((max_jet), dtype = bool)     # preallocate
                        for i in range(branches):
                            valid = pred[i][(pred[i] >= 0) & (pred[i] < max_jet)]    # set T for all non-negative integers
                            used_jet_idx[valid] = False     # set False for all used jet_idx for all branches
                        available_mask = available_jets_mask & (~used_jet_idx)              # True if jet is available and not used
                        # available_jets_idx = np.argwhere(available_mask)[0].tolist()        # map to numbers 0-9
                        available_jets_idx = np.argwhere(available_mask).flatten().tolist() # map to numbers 0-9
                        for idx, jet_idx in enumerate(targ):
                            if jet_idx < 0:
                                targ[idx] = available_jets_idx[-1 - idx]      # NOTE: Negative indices (-1) in target are filled with random available jet indices (probably does not respect physics)
                targ_arr[branch, event, :] = targ       # save unpadded-padded idx
                for idx in pred:
                    features_arr[event, k, branch, idx, :] = jet_data[event, idx, :]   # retrieve features from jet_data using pred idx
        return truth, targ_arr, true_masks, features_arr
