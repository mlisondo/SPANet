from typing import Dict
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from spanet.options import Options
from spanet.dataset.evaluator import SymmetricEvaluator
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork

class TopKMatchDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):                     # combined chat and ML tutorial, not sure if this part is correct
        e = self.entries[idx]
        feat = torch.from_numpy(e["features"]).float()
        score = torch.tensor(e["score"], dtype=torch.float32)
        label = torch.tensor(e["k_rank"], dtype=torch.long)
        return feat, score, label


class JetReconstructionValidation(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetReconstructionValidation, self).__init__(options, torch_script)
        self.evaluator = SymmetricEvaluator(self.training_dataset.event_info)

    def topk_pipeline(self, batch, batch_idx) -> Dict[str, np.float32]:
        '''
        > check target validation in jet prediction →if no return error
        > if prediction contains true val →get feature vector
        > return jet indices →should be object: source.data →contains 5 features
            ▶ mass, pt , η, ϕ, btag


        1) Unpack inputs and raw features.
        2) Run your top-k assignment extractor.
        3) Iterate over every event & branch, validating that a true branch exists.
        4) Scan your k candidate assignments, compare to ground truth.
        5) On match, pull out the jets' 5-dim feature vectors and package everything into a result.
        6) Raise an error if no match is found.
        '''
                    # B = batch size = 32
                    # J = max jets   = 10
                    # F =  feature   = 5  = [mass, pt, η, ϕ, btag]
                    # P = partons    = 3
                    # k = hypothesis
        sources, _, targets, _, _ = batch
        # targets[i]: (indices, masks)      (i=0 top; i=1 antitop)
            # indices: (B, P)       3 jet indecies for particle i
            # masks:   (B,)         bool whether particle i existed

        jet_data, jet_mask = sources[0]         # do i use jet_mask??
        # jet_data: (B, J, F), jet_mask: (B, J)

        jet_preds, particle_scores, _, _ = self.predict(sources)
        # jet_preds[i]: (B, P, k) particle_scores[i]: (B,)       (i=0 top; i=1 antitop)

        batch_size  = jet_data.size(0)
        num_targets = len(targets)            # number of branches (2 for ttbar)
        k_len       = jet_preds[0].shape[2]

        stacked_targets = np.zeros(num_targets, dtype=object)  # will hold (B,P) arrays
        stacked_masks   = np.zeros((num_targets, batch_size),  dtype=bool)

        for i, (t_idx, t_mask) in enumerate(targets):
            stacked_targets[i] = t_idx.detach().cpu().numpy()   # shape: (B, P)
            stacked_masks[i]   = t_mask.detach().cpu().numpy()  # shape: (B,)

        for i, decoder in enumerate(self.branch_decoders):
            groups = decoder.permutation_indices   # list of symmetric parton‐index lists
            preds  = jet_preds[i]                  # (B, P, k)
            targ   = stacked_targets[i]            # (B, P)

            for grp in groups:
                if len(grp) > 1:
                    # sort truth and each hypothesis along that parton‐axis
                    targ[:, grp]     = np.sort(targ[:, grp], axis=1)
                    preds[:, grp, :] = np.sort(preds[:, grp, :], axis=1)

        topk_dataset = []

        for branch in range(num_targets):
            for event in range(batch_size):
                true_idx = stacked_targets[branch][event] #  targets[branch].indices[event] ?
                if not stacked_masks[branch,event]: # check if particle actully exists
                    continue
                for k in range(k_len):
                    if jet_preds[branch][event,:,k] == true_idx:
                        features = jet_data[event, true_idx, :]
                        score = particle_scores[branch][event]
                        topk_dataset.append({
                            "branch": branch,
                            "event": event,
                            "k_rank": k,
                            "jet_indices": true_idx,
                            "features": features,
                            "score": score
                        })
        
        dataset = TopKMatchDataset(topk_dataset)
        loader  = DataLoader(dataset, batch_size=32)
        return loader

