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

    def best_truth_permutation(
        self,
        pred_sorted: torch.Tensor,   # (E, K, B, p)
        truth_sorted: torch.Tensor,  # (E, B, p)
        true_masks: torch.Tensor,    # (E, B)
        pad_val: int = -1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Finds the best permutation of branches in the truth to match the prediction.
        
        Returns:
            permuted_truth: (E, B, p)
            permuted_mask:  (E, B)
            pred_truth:     (E, K, B) — best-matching mask-aware equality
        """
        E, K, B, p = pred_sorted.shape
        perms = list(itertools.permutations(range(B)))
        num_perms = len(perms)

        best_truth = torch.empty_like(truth_sorted)
        best_mask  = torch.empty_like(true_masks)
        best_score = torch.full((E, K), -1, dtype=torch.long, device=pred_sorted.device)
        best_pred_truth = torch.zeros((E, K, B), dtype=torch.bool, device=pred_sorted.device)

        for perm in perms:
            # Apply permutation to truth and mask
            perm = torch.tensor(perm, device=pred_sorted.device)
            truth_perm = truth_sorted[:, perm, :]      # (E, B, p)
            mask_perm  = true_masks[:, perm]           # (E, B)

            # Expand for matching
            truth_expand = truth_perm.unsqueeze(1)     # (E, 1, B, p)
            valid = (truth_expand != pad_val)          # (E,1,B,p)

            # Compare prediction with permuted truth
            eq = (pred_sorted == truth_expand) | (~valid)
            pred_truth = eq.all(dim=-1)                # (E,K,B)

            # Score: count matches where truth_mask is True and pred_truth is True
            score = (pred_truth & mask_perm.unsqueeze(1)).sum(dim=-1)  # (E,K)

            # Update best match
            update = score > best_score                # (E,K)
            update_mask = update.unsqueeze(-1)         # (E,K,1)

            best_score = torch.where(update, score, best_score)
            best_pred_truth = torch.where(update_mask, pred_truth, best_pred_truth)

            # Store best permutation of truth/mask per event
            for e in range(E):
                for k in range(K):
                    if update[e, k]:
                        best_truth[e] = truth_perm[e]
                        best_mask[e]  = mask_perm[e]

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
        canon_truth, canon_masks, pred_truth = self.best_truth_permutation(
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

        return pred_truth, class_truth, features_arr, canon_truth, canon_masks

    @torch.no_grad()
    def topk_data(self, batch):
        sources, _, targets, _, _ = batch
        jet_data, _ = sources[0]  # (E,Njets,F)
    
        raw_preds, *_ = self.predict(sources)  # list[B] of (E,K,p_i)
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
    
        pred_truth, class_truth, features_arr, canon_truth, canon_masks = self._topk_core(
            jet_data, jet_preds_tensor, true_idx, true_masks
        )

        super_true_event_idx = torch.nonzero(true_masks.all(dim=0)).squeeze(1)[:2]

        true_event_idx = torch.nonzero(class_truth[:, 0]).squeeze(1)[:2]

        false_event_idx = torch.nonzero(~class_truth[:, 0]).squeeze(1)[:2]

        one_one = torch.cat([super_true_event_idx, true_event_idx, false_event_idx])  

        probe(sources[0], "sources[0]")
        probe(jet_data, "jet_data")
        probe(jet_preds_tensor, "jet_preds_tensor")
        probe(true_idx, "true_idx")
        probe(true_masks, "true_masks")
        probe(pred_truth, "pred_truth")
        probe(class_truth, "class_truth")
        probe(features_arr, "features_arr")
        probe(jet_data, "jet_data")
        probe(jet_preds_tensor, "jet_preds_tensor")
        probe(canon_truth, "canon_truth")
        probe(canon_masks, "canon_masks")

        for e in one_one:
            print(f"\n===== EVENT {int(e)} =====")

            print("Jet data for events:")
            print(jet_data[e])

            print("jet_preds_tensor:")
            print(jet_preds_tensor[e])

            print("true_idx:")
            print(true_idx[:, e])

            print("canon_truth:")
            print(canon_truth[:, e])

            print("true_masks:")
            print(true_masks[:, e])

            print("canon_masks")
            print(canon_masks[:, e])

            print("pred_truth matrix (K x B):")
            print(pred_truth[e])

            print("class_truth row:")
            print(class_truth[e])

            print("feature for selected events:")
            print(features_arr[e])

            print("=" * 30)

        raise RuntimeError("Debug break")

        return pred_truth, canon_masks, features_arr, class_truth, canon_truth, jet_preds_tensor




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