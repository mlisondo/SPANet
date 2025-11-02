import torch
import itertools
import torch.nn.functional as F

from spanet.options import Options
from spanet.dataset.evaluator import SymmetricEvaluator
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork

class JetSecondaryLoader(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetSecondaryLoader, self).__init__(options, torch_script)
        self.evaluator = SymmetricEvaluator(self.training_dataset.event_info)
        self.options = options

    def slot_jet_marginals(S: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        S: (E, J, J, ..., J)  # p slot axes after batch dim
        returns (E, p, J)     # per-slot, per-jet 'jet_scores'
        """
        p = S.dim() - 1
        # Detect probs vs logits; convert probs -> log for stable logsumexp
        if S.min() >= 0 and S.max() <= 1.0001:
            Slog = S.clamp_min(eps).log()
        else:
            Slog = S
        outs = []
        for s in range(p):
            reduce_dims = tuple(1 + d for d in range(p) if d != s)
            outs.append(torch.logsumexp(Slog, dim=reduce_dims))  # (E, J)
        return torch.stack(outs, dim=1)  # (E, p, J)

    @torch.no_grad()
    def best_truth_permutation_vectorized(
        self,
        pred_sorted: torch.Tensor,   # (E, K, B, p)
        truth_sorted: torch.Tensor,  # (E, B, p)
        true_masks: torch.Tensor,    # (E, B)
        pad_val: int = -1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        E, K, B, p = pred_sorted.shape
        device = pred_sorted.device
    
        # All permutations of branch indices (P,B)
        perm_idx = torch.tensor(list(itertools.permutations(range(B))),
                                device=device, dtype=torch.long)      # (P,B)
        P = perm_idx.shape[0]
    
        # Apply permutations: (E,P,B,p), (E,P,B)
        truth_perm = truth_sorted[:, perm_idx, :]                     # (E,P,B,p)
        mask_perm  = true_masks[:,  perm_idx]                         # (E,P,B)
    
        # Mask-aware equality across p
        pred_e  = pred_sorted.unsqueeze(1)                            # (E,1,K,B,p)
        truth_e = truth_perm.unsqueeze(2)                             # (E,P,1,B,p)
        valid   = (truth_e != pad_val)
        eq      = (pred_e == truth_e) | (~valid)                      # (E,P,K,B,p)
        pred_truth_all = eq.all(dim=-1)                               # (E,P,K,B) bool
    
        # Scores per perm and (E,K)
        score = (pred_truth_all & mask_perm.unsqueeze(2)).sum(dim=-1) # (E,P,K) int64
    
        # For each (E,K): first permutation index achieving the max score
        best_score_e_k, _ = score.max(dim=1)                          # (E,K)
        first_is_max = (score == best_score_e_k.unsqueeze(1))         # (E,P,K)
        idx_first = first_is_max.int().argmax(dim=1)                  # (E,K) earliest index with max
    
        # Event-level permutation used for best_truth/mask:
        # choose the permutation whose "first-hit" index is latest across K
        idx_event = idx_first.max(dim=-1).values                      # (E,)
    
        # Outputs
        best_truth = truth_perm[torch.arange(E, device=device), idx_event]  # (E,B,p)
        best_mask  = mask_perm[ torch.arange(E, device=device), idx_event]  # (E,B)
    
        # best_pred_truth per K at that K's own best permutation
        gather_idx = idx_first.view(E, 1, K, 1).expand(-1, 1, -1, pred_truth_all.size(-1))
        best_pred_truth = pred_truth_all.gather(dim=1, index=gather_idx).squeeze(1)  # (E,K,B)
    
        return best_truth, best_mask, best_pred_truth
    
    @torch.compile(dynamic=True)
    def _topk_core(
        self,
        jet_data: torch.Tensor,          # (E, Njets, F)
        jet_preds_tensor: torch.Tensor,  # (E, K, B, p_max)
        true_idx_tensor: torch.Tensor,   # (B, E, p_max)  – pad = -1
        true_masks_tensor: torch.Tensor, # (B, E)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Permutation-invariant branch match using best permutation of truth.
        Returns:
            pred_truth:    (E, K, B)
            class_truth:   (E, K)
            features_arr:  (E, K, B, p_max, F)
        """
        PAD = -1
        E, K, B, p_max = jet_preds_tensor.shape
        _, Njets, Fdim = jet_data.shape

        # Step 1: Sort jets within each branch
        pred_sorted  = torch.sort(jet_preds_tensor, dim=-1).values                         # (E, K, B, p)
        truth_tensor = true_idx_tensor.permute(1, 0, 2)                                    # (E, B, p)
        truth_sorted = torch.sort(truth_tensor, dim=-1).values                             # (E, B, p)
        mask_matrix  = true_masks_tensor.permute(1, 0)                                     # (E, B)

        # Step 2: Canonicalize truth via best permutation to match each prediction
        canon_idx, canon_masks, pred_truth = self.best_truth_permutation_vectorized(
            pred_sorted, truth_sorted, mask_matrix, PAD
        )

        # Step 3: Extract features for each predicted jet index
        flat_idx = jet_preds_tensor.reshape(E, K * B * p_max).long().clamp(min=0, max=Njets - 1)
        features_arr = jet_data.gather(
            dim=1, index=flat_idx.unsqueeze(-1).expand(-1, -1, Fdim)
        ).view(E, K, B, p_max, Fdim)

        # Step 4: Derive class-level truth from branch match and mask
        has_rec     = canon_masks.any(dim=1, keepdim=True)                 # (E, 1)
        mask_exp    = canon_masks.unsqueeze(1)                             # (E, 1, B)
        branch_ok   = (pred_truth == mask_exp).all(dim=2)                  # (E, K)
        class_truth = branch_ok & has_rec.expand_as(branch_ok)            # (E, K)

        return pred_truth, class_truth, features_arr, canon_idx, canon_masks

    @torch.no_grad()
    def topk_data(self, batch):
        sources, _, targets, _, _ = batch
        jet_data = sources[0][0]  # (E,Njets,F)
        jet_mult = sources[0][1]  # (E,Njets) bool  for ttbar (Njets) = 10, true if jet is valid
    
        raw_preds, particle_scores, *_ = self.predict(sources)  # list[B] of (E,K,p_i)

        jet_preds_tensor = torch.stack(
            [torch.as_tensor(p, device=jet_data.device).permute(0, 2, 1)
             for p in raw_preds],
            dim=2
        )  # (E,K,B,p_max)
        p_max = jet_preds_tensor.shape[-1]
    
        true_idx, true_masks = [], []
        for idx_t, m in targets:
            if idx_t.shape[1] < p_max:
                idx_t = F.pad(idx_t, (0, p_max - idx_t.shape[1]), value=-1)
            true_idx.append(idx_t.to(jet_data.device))
            true_masks.append(m.to(jet_data.device))
    
        true_idx   = torch.stack(true_idx)   # (B,E,p_max)
        true_masks = torch.stack(true_masks) # (B,E)

        pred_truth, class_truth, features_arr, canon_idx, canon_masks = self._topk_core(
            jet_data, jet_preds_tensor, true_idx, true_masks
        )
        canon_idx = canon_idx.permute(1, 0, 2)
        canon_masks = canon_masks.permute(1, 0)

        # JET SCORES
        with torch.no_grad():
            outputs = self.forward(sources)
        probe(outputs, "outputs")

        scores = [S.to(jet_data.device) for S in outputs.assignments] # [B, E, J, J, J]
        probe(scores, "scores")

        scores = scores.permute(1, 0)
        probe(scores, "scores")

        # features_arr = torch.cat([features_arr, jet_scores.unsqueeze(-1)], dim=-1)

        # super_true_event_idx = torch.nonzero(true_masks.all(dim=0)).squeeze(1)[:2]

        # true_event_idx = torch.nonzero(class_truth[:, 0]).squeeze(1)[:2]

        # false_event_idx = torch.nonzero(~class_truth[:, 0]).squeeze(1)[:2]

        # one_one = torch.cat([super_true_event_idx, true_event_idx, false_event_idx])  

        # probe(sources[0], "sources[0]")
        # probe(jet_data, "jet_data")
        # probe(jet_preds_tensor, "jet_preds_tensor")
        # probe(jet_mult, "jet_mult")
        # # probe(true_idx, "true_idx")
        # # probe(true_masks, "true_masks")
        # # probe(pred_truth, "pred_truth")
        # # probe(class_truth, "class_truth")
        # probe(features_arr, "features_arr")
        # probe(jet_data, "jet_data")
        # # probe(canon_idx, "canon_idx")
        # # probe(canon_masks, "canon_masks")
        # probe(particle_scores, "particle_scores")

        # for e in one_one:
        #     print(f"\n===== EVENT {int(e)} =====")

        #     print("Jet data for events:")
        #     print(jet_data[e])

        #     print("Jet options for events:")
        #     print(jet_mult[e])

        #     print("jet_preds_tensor:")
        #     print(jet_preds_tensor[e])

        #     print("particle scores:")
        #     print(particle_scores[e])

        #     # print("true_idx:")
        #     # print(true_idx[:, e])

        #     # print("canon_idx:")
        #     # print(canon_idx[:, e])

        #     # print("true_masks:")
        #     # print(true_masks[:, e])

        #     # print("canon_masks")
        #     # print(canon_masks[:, e])

        #     # print("pred_truth matrix (K x B):")
        #     # print(pred_truth[e])

        #     # print("class_truth row:")
        #     # print(class_truth[e])

        #     print("feature for selected events:")
        #     print(features_arr[e])

        #     print("=" * 30)

        raise RuntimeError("Debug break")

        return pred_truth, canon_masks, features_arr, class_truth, canon_idx, jet_preds_tensor, jet_mult




def probe(o, name=None):
    obj = type(o)
    header = f"Object '{name}'"
    print(f"\n{header}: {obj.__module__}.{obj.__name__}")

    # NumPy-style introspection
    if hasattr(o, 'shape'):
        print(f"shape: {o.shape}")
    if hasattr(o, 'ndim'):
        print(f"ndim: {o.ndim}")
    if hasattr(o, 'dtype'):
        print(f"dtype: {o.dtype}")

    # size attribute
    if hasattr(o, 'size') and not callable(o.size):
        print(f"size: {o.size}")

    # Pythonic length
    try:
        print(f"len: {len(o)}")
    except Exception:
        pass

    # Recursive descent into lists
    try:
        if isinstance(o, (list, tuple)):
            for idx, item in enumerate(o):
                probe(item, f"{name}[{idx}]")
    except Exception:
        pass

    # PyTorch tensors
    if isinstance(o, torch.Tensor):
        print(f"shape: {tuple(o.size())}")
        print(f"dtype: {o.dtype}")
        print(f"numel: {o.numel()}")

        print(f"shape: {tuple(o.size())}")
        print(f"dtype: {o.dtype}")
        print(f"numel: {o.numel()}")
        print(f"device: {o.device}")
