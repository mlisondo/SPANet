import torch

from spanet.options import Options
from spanet.dataset.evaluator import SymmetricEvaluator
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork
import torch.nn.functional as F

class JetSecondaryLoader(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetSecondaryLoader, self).__init__(options, torch_script)
        self.evaluator = SymmetricEvaluator(self.training_dataset.event_info)
        self.options = options
        self._topk_core = torch.compile(self._topk_core, dynamic=True)   # or mode="max-autotune"
    


    def _sort_ignore_pad(self, x: torch.Tensor, pad_val: int, high_val: int) -> torch.Tensor:
        """Sort last dim but shove pad_val to the end."""
        sentinel = torch.full_like(x, high_val)
        x_tmp    = torch.where(x == pad_val, sentinel, x)
        x_sorted, _ = x_tmp.sort(dim=-1)
        return torch.where(x_sorted == sentinel, torch.full_like(x_sorted, pad_val), x_sorted)
    
    @staticmethod
    def _topk_core(
        self,
        jet_data: torch.Tensor,          # (E, Njets, F)
        jet_preds_tensor: torch.Tensor,  # (E, K, B, p_max)
        true_idx_tensor: torch.Tensor,   # (B, E, p_max)  – pad = -1
        true_masks_tensor: torch.Tensor, # (B, E)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Permutation-invariant branch match without reordering branches.
        """
        PAD = -1
        E, K, B, p_max = jet_preds_tensor.shape
        _, Njets, Fdim = jet_data.shape
    
        # 1) Sort jets within each branch / truth, ignore PAD
        pred_sorted = self._sort_ignore_pad(jet_preds_tensor, pad_val=PAD, high_val=Njets + 1)  # (E,K,B,p)
        truth       = true_idx_tensor.permute(1, 0, 2)                                     # (E,B,p)
        truth_sorted = self._sort_ignore_pad(truth, pad_val=PAD, high_val=Njets + 1)            # (E,B,p)
    
        # 2) Compare branch-wise, ignoring padded truth locations
        valid_truth_mask = (truth_sorted != PAD).unsqueeze(1)                               # (E,1,B,p)
        eq = (pred_sorted == truth_sorted.unsqueeze(1)) | (~valid_truth_mask)
        pred_truth = eq.all(dim=-1)                                                         # (E,K,B)
    
        # 3) Gather features with original (unsorted) prediction indices
        flat_idx = jet_preds_tensor.reshape(E, K * B * p_max).long()
        flat_idx = flat_idx.clamp_(min=0, max=Njets - 1)  # safe clamp instead of in-place scatter
        gathered = jet_data.gather(
            1, flat_idx.unsqueeze(-1).expand(-1, -1, Fdim)
        ).view(E, K, B, p_max, Fdim)
    
        # 4) Hypothesis-level truth: “Did any branch match?” ↔ “Is there a target?”
        mask_matrix = true_masks_tensor.permute(1, 0)             # (E,B)
        any_match   = pred_truth.any(dim=1)                       # (E,B)
        event_ok    = (any_match == mask_matrix)                  # (E,B)
    
        # Preserve old API: (E,K) shaped class_truth
        class_truth = event_ok.all(dim=1, keepdim=True).expand(-1, K)
    
        return pred_truth, class_truth, gathered
    
    # Compile

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
    
        pred_truth, class_truth, features_arr = self._topk_core(
            jet_data, jet_preds_tensor, true_idx, true_masks
        )

        probe(batch, "batch")
        probe(sources, "sources")
        probe(targets, "targets")
        probe(jet_data, "jet_data")
        probe(raw_preds, "raw_preds")
        probe(jet_preds_tensor, "jet_preds_tensor")
        probe(p_max, "p_max")
        probe(true_idx, "true_idx")
        probe(true_masks, "true_masks")
        probe(pred_truth, "pred_truth")
        probe(class_truth, "class_truth")
        probe(features_arr, "features_arr")


        print("jet_preds_tensor[:4]", jet_preds_tensor[:4])
        print()
        print()
        print("true_idx[0][:4]", true_idx[0][:4])
        print()
        print()
        print("true_idx[1][:4]", true_idx[1][:4])
        print()
        print()
        print("pred_truth[:4]", pred_truth[:4])
        print()
        print()
        print("true_masks[0][:4]", true_masks[0][:4])
        print()
        print()
        print("true_masks[1][:4]", true_masks[1][:4])
        print()
        print()
        print("class_truth[:4]", class_truth[:4])
        print()
        print()

        raise RuntimeError("Debug break")

        return pred_truth, true_masks, features_arr, class_truth




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
