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


    def _sort_ignore_pad(x, pad_val, high_val):
        # Push pad_val to the end, sort, bring pad_val back.
        sentinel = torch.full_like(x, high_val)
        x_tmp     = torch.where(x == pad_val, sentinel, x)
        x_sorted, _ = x_tmp.sort(dim=-1)
        return torch.where(x_sorted == sentinel, torch.full_like(x_sorted, pad_val), x_sorted)
    
    # compile with dynamic shapes
    _sort_ignore_pad = torch.compile(_sort_ignore_pad, dynamic=True)

    @staticmethod
    def _topk_core(
        jet_data: torch.Tensor,        # (E, Njets, F)
        jet_preds_tensor: torch.Tensor,# (E, K, B, p_max)
        true_idx_tensor: torch.Tensor, # (B, E, p_max)  – pad = -1
        true_masks_tensor: torch.Tensor,# (B, E)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Canonical, permutation‑invariant label/feature extractor."""
        E, K, B, p_max = jet_preds_tensor.shape
        _, Njets, Fdim = jet_data.shape
        PAD = -1

        # ---- 1. jet‑level sort (pad last) -----------------------
        pred_sorted = JetSecondaryLoader._sort_ignore_pad(
            jet_preds_tensor, pad_val=PAD, high_val=Njets + 1
        )                                              # (E,K,B,p_max)

        truth_sorted = JetSecondaryLoader._sort_ignore_pad(
            true_idx_tensor.permute(1, 0, 2),          # (E,B,p_max)
            pad_val=PAD, high_val=Njets + 1
        )                                              # (E,B,p_max)

        # ---- 2. branch‑level canonical order -------------------
        weight = (Njets + 2) ** torch.arange(
            p_max, device=jet_data.device, dtype=pred_sorted.dtype
        )
        key_pred  = (pred_sorted  + 1).mul(weight).sum(-1)  # (E,K,B)
        key_truth = (truth_sorted + 1).mul(weight).sum(-1)  # (E,B)

        order_pred  = key_pred.argsort(2)                   # branch axis = 2
        order_truth = key_truth.argsort(1)                  # axis = 1

        pred_sorted  = torch.gather(
            pred_sorted, 2, order_pred.unsqueeze(-1).expand_as(pred_sorted)
        )
        truth_sorted = torch.gather(
            truth_sorted, 1, order_truth.unsqueeze(-1).expand_as(truth_sorted)
        )

        # ---- 3. branch‑wise equality ---------------------------
        valid = (truth_sorted != PAD).unsqueeze(1)           # (E,1,B,p_max)
        eq = (pred_sorted == truth_sorted.unsqueeze(1)) | (~valid)
        pred_truth = eq.all(-1)                              # (E,K,B)

        # ---- 4. gather jet features (safe indices) ------------
        flat_idx = pred_sorted.reshape(E, K * B * p_max).long()
        flat_idx[flat_idx < 0]      = 0           # pad → 0
        flat_idx[flat_idx >= Njets] = Njets - 1   # overflow → last jet

        gathered = jet_data.gather(
            1, flat_idx.unsqueeze(-1).expand(-1, -1, Fdim)
        ).view(E, K, B, p_max, Fdim)

        # ---- 5. hypothesis‑level correctness ------------------
        mask_matrix = true_masks_tensor.permute(1, 0)        # (E,B)
        all_match   = pred_truth.all(2)                      # (E,K)
        class_truth = all_match == mask_matrix.any(1, keepdim=True)

        return pred_truth, class_truth, gathered
        
    # compile with dynamic shapes
    _topk_core = torch.compile(_topk_core, dynamic=True)
    
    @torch.no_grad()
    def topk_data(self, batch):
        sources, _, targets, _, _ = batch
        jet_data, _ = sources[0]  # (E,Njets,F)
    
        # PRE-PROCESS #
        raw_preds, *_ = self.predict(sources)  # list[B] of (E,K,p_i) (no guarantee they are sorted)
        jet_preds_tensor = torch.stack(
            [torch.as_tensor(p, device=jet_data.device).permute(0, 2, 1)
             for p in raw_preds],
            dim=2
        )  # (E,K,B,p_max)
        p_max = jet_preds_tensor.shape[-1]
    
        true_idx, true_masks = [], []
        for idx_t, m in targets:
            # pad to p_max if needed
            if idx_t.shape[1] < p_max:
                idx_t = F.pad(idx_t, (0, p_max - idx_t.shape[1]), value=-1)
            true_idx.append(idx_t.to(jet_data.device))
            true_masks.append(m.to(jet_data.device))
    
        true_idx   = torch.stack(true_idx)   # (B,E,p_max)
        true_masks = torch.stack(true_masks) # (B,E)
    
        pred_truth, class_truth, features_arr = self.__class__._topk_core(
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