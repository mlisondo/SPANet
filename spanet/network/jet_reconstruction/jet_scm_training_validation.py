import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List, Dict
from spanet.options import Options
from spanet.network.jet_reconstruction.jet_scm_pipeline import JetSecondaryLoader
from spanet.dataset.types import Batch

tcompile = torch.compile

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,   # (N, S, E)
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        # x: (N, S, E)
        return self.enc(x)


class ClassifierTransformerHead(nn.Module):
    """
    Inputs:
      features_arr: (N, K, B, J, F)
    Outputs:
      logits: (N, real_K) where real_K = B*K - 1
      token_scores: (N, K) optional per-K score for argmax monitoring
    """
    def __init__(self, branch_dim: int, jets: int, feats: int,
                 class_embed_dim: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        self.branch_dim = branch_dim
        self.token_in_dim = branch_dim * jets * feats
        self.proj = nn.Linear(self.token_in_dim, class_embed_dim)
        self.tr = SimpleTransformerEncoder(class_embed_dim, nhead, num_layers, dropout)
        # Per-token predicts branch_dim logits
        self.head = nn.Linear(class_embed_dim, branch_dim)

    def forward(self, features_arr):
        N, K, B, J, Fdim = features_arr.shape
        
        probe(features_arr, "features_arr (classifier input)")

        # tokens: (N, K, B*J*F)
        tokens = features_arr.reshape(N, K, B * J * Fdim)
        x = self.proj(tokens)           # (N, K, E)
        x = self.tr(x)                  # (N, K, E)
        per_token_branch = self.head(x) # (N, K, B)

        probe(tokens, "tokens")


        # For logging a single best-K index: score each token by its best branch logit
        token_scores, _ = per_token_branch.max(dim=-1)  # (N, K)

        logits = per_token_branch.reshape(N, K * B)
        logits = logits[:, :real_K]
        
        return logits, token_scores


class MaskerTransformerHead(nn.Module):
    """
    Inputs:
      x: (N, B, J, F)
    Outputs:
      logits: (N, B)
    """
    def __init__(self, feats: int, mask_embed_dim: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(feats, mask_embed_dim)
        self.tr = SimpleTransformerEncoder(mask_embed_dim, nhead, num_layers, dropout)
        self.head = nn.Linear(mask_embed_dim, 1)

    def forward(self, x):
        N, B, J, Fdim = x.shape
        # reshape to process each branch independently: batch B groups
        x = x.reshape(N * B, J, Fdim)    # (N*B, J, F)
        x = self.proj(x)                 # (N*B, J, E)
        x = self.tr(x)                   # (N*B, J, E)
        x = x.mean(dim=1)                # mean pool over jets -> (N*B, E)
        logits = self.head(x).squeeze(-1).reshape(N, B)  # (N, B)
        return logits


class SCM_Training_Val(JetSecondaryLoader):
    def __init__(self, options: Options, torch_script: bool = False):
        super(SCM_Training_Val, self).__init__(options, torch_script)
        self.options = options

        # New transformer config with sensible defaults if missing
        self.class_embed_dim = options.class_embed_dim
        self.mask_embed_dim  = options.mask_embed_dim
        self.class_nhead     = options.class_nhead
        self.mask_nhead      = options.mask_nhead
        self.class_layers    = options.class_layers
        self.mask_layers     = options.mask_layers
        self.tr_dropout      = 0.0
        self.mask_reduction  = "any"  # "any" | "mean" | "max"

        B = self.options.branch_dim
        K = self.options.k
        J = self.options.jet_max_dim
        Fdim = self.options.features_dim
        self.real_K = B * K - 1

        # --- Transformer heads (no positional encodings) ---
        self.classifier = ClassifierTransformerHead(
            branch_dim=B, jets=J, feats=Fdim,
            class_embed_dim=self.class_embed_dim,
            nhead=self.class_nhead, num_layers=self.class_layers,
            dropout=self.tr_dropout,
        )
        self.masker = MaskerTransformerHead(
            feats=Fdim, mask_embed_dim=self.mask_embed_dim,
            nhead=self.mask_nhead, num_layers=self.mask_layers,
            dropout=self.tr_dropout,
        )

        # Compile
        self.classifier = tcompile(self.classifier, dynamic=True)
        self.masker     = tcompile(self.masker,    dynamic=True)

        # Imbalance / focal
        self.pos_weight_cap = 1000.0
        self.use_focal_masker = "use_focal_masker"
        self.focal_alpha_pos = 0.7
        self.focal_gamma = 2.0

    def _compiled_core(self, features_arr, pred_truth, class_truth):
        # features_arr: (N, K, B, J, F)
        N, K, B, J, Fdim = features_arr.shape

        class_logits, token_scores = self.classifier(features_arr)  # (N, real_K), (N, K)

        class_loss, has_truth, num_pos, ce_random_baseline = self._multi_positive_ce(class_logits, class_truth)

        rows = torch.arange(N, device=class_logits.device)
        pred_k = torch.argmax(token_scores, dim=1)  # best hypothesis index by max token score
        if has_truth.any():
            # Build a K-hot truth over hypotheses: treat a class index block per K of size B
            # A token is considered "true" if any of its B branch entries is true in class_truth.
            # class_truth is (N, real_K) corresponding to [k0:b0..bB-1, k1:..., ...] minus the last slot.
            # Rebuild a padded view to (N, K, B) with a trailing zero column to restore full K*B.
            padded = torch.zeros(N, K * B, device=class_logits.device, dtype=class_truth.dtype)
            padded[:, :self.real_K] = class_truth
            truth_kb = padded.view(N, K, B)
            truth_k = truth_kb.any(dim=2)  # (N, K)
            top1_acc_truth = truth_k[rows[has_truth], pred_k[has_truth]].float().mean()
            num_pos_mean = num_pos[has_truth].float().mean()
        else:
            top1_acc_truth = torch.tensor(0.0, device=class_logits.device)
            num_pos_mean = num_pos.float().mean()
        has_truth_frac = has_truth.float().mean()

        # ---- Masker via Transformer over jets per branch ----
        # Reduce targets from (N, K, B) -> (N, B)
        t_kb = pred_truth.float()
        if self.mask_reduction == "any":
            t_branch = t_kb.bool().any(dim=1).float()
        elif self.mask_reduction == "mean":
            t_branch = t_kb.float().mean(dim=1)  # in [0,1]
        elif self.mask_reduction == "max":
            t_branch = t_kb.float().amax(dim=1)
        else:
            raise ValueError(f"Unknown mask_target_reduction: {self.mask_reduction}")

        # Build masker inputs: (N, B, J, F)
        # Take the first K hypothesis (or we could average features over K). To stay order-free, use mean over K.
        features_branch = features_arr.mean(dim=1)  # (N, B, J, F)
        logits_branch = self.masker(features_branch)  # (N, B)

        # Imbalance stats per branch (using reduced targets)
        t = t_branch
        pos_per_branch = t.sum(dim=0)  # (B,)
        tot_per_branch = torch.tensor(N, device=t.device, dtype=t.dtype)
        neg_per_branch = tot_per_branch - pos_per_branch
        eps = torch.finfo(t.dtype).eps
        pos_weight_b = (neg_per_branch / (pos_per_branch + eps)).clamp(max=self.pos_weight_cap)

        mask_loss = self.focal_bce_with_logits(
            logits_branch, t_branch,
            alpha_pos=self.focal_alpha_pos,
            gamma=self.focal_gamma,
            reduction="mean",
        )

        pos_rate = (pos_per_branch.sum() / (N * B)).to(logits_branch.dtype)
        avg_pos_weight = pos_weight_b.mean()

        return (
            class_loss, mask_loss, top1_acc_truth,
            has_truth_frac, num_pos_mean, ce_random_baseline,
            pos_rate, avg_pos_weight
        )


    # Multi-positive classifier loss
    @staticmethod
    def _multi_positive_ce(class_logits: torch.Tensor, class_truth: torch.Tensor):
        N, C = class_logits.shape
        pos_mask = class_truth.bool()
        has_truth = pos_mask.any(dim=1)

        log_probs = torch.log_softmax(class_logits, dim=1)

        # --- DEBUG SHAPES BEFORE masked_fill ---
        probe(class_logits, "class_logits")
        probe(class_truth, "class_truth")
        probe(log_probs, "log_probs")
        probe(pos_mask, "pos_mask")

        lp_masked = log_probs.masked_fill(~pos_mask, float("-inf"))

        pos_lse = torch.logsumexp(lp_masked, dim=1)            # [N]
        num_pos = pos_mask.sum(dim=1)                           # [N]
        num_pos_clamped = num_pos.clamp_min(1).to(log_probs.dtype)

        loss_vec = -(pos_lse - torch.log(num_pos_clamped))
        loss = loss_vec[has_truth].mean() if has_truth.any() else loss_vec.mean()

        with torch.no_grad():
            ce_baseline = torch.log(torch.tensor(C, dtype=log_probs.dtype, device=log_probs.device)) \
                        - torch.log(num_pos_clamped)
            ce_baseline = ce_baseline[has_truth].mean() if has_truth.any() else ce_baseline.mean()

        return loss, has_truth, num_pos, ce_baseline

    # Focal loss to bias toward positive class and difficult examples
    @staticmethod
    def focal_bce_with_logits(logits, targets, alpha_pos=0.25, gamma=2.0, reduction="mean"):
        p = torch.sigmoid(logits)
        pt = torch.where(targets.bool(), p, 1 - p)  # p_t
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        alpha_t = torch.where(
            targets.bool(),
            torch.as_tensor(alpha_pos, device=logits.device, dtype=logits.dtype),
            torch.as_tensor(1 - alpha_pos, device=logits.device, dtype=logits.dtype),
        )
        loss = alpha_t * (1 - pt).pow(gamma) * bce

        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        return loss

    _compiled_core = tcompile(_compiled_core, dynamic=True)

    def forward_scm(self, batch):
        pred_truth, true_masks, features_arr, class_truth, true_idx, jet_preds_tensor = self.topk_data(batch)

        probe(features_arr, "features_arr")
        probe(class_truth, "class_truth")
        
        print(f"[CONFIG] K={self.options.k}, B={self.options.branch_dim}, real_K={self.real_K}")

        return self._compiled_core(features_arr, pred_truth, class_truth)

    def training_step(self, batch: Batch, batch_idx: int):
        self.on_train_epoch_start()
        (
            class_loss, mask_loss, top1_acc_truth,
            has_truth_frac, num_pos_mean, ce_random_baseline,
            pos_rate, avg_pos_weight
        ) = self.forward_scm(batch)

        total_loss = class_loss + mask_loss

        self.log('train_classifier_loss', class_loss)
        self.log('train_masker_loss', mask_loss)
        self.log('train_total_loss', total_loss)
        self.log('train_top1_acc_truth', top1_acc_truth)
        self.log('train_has_truth_frac', has_truth_frac)
        self.log('train_num_pos_mean', num_pos_mean)
        self.log('train_ce_random_baseline', ce_random_baseline)
        self.log('train_mask_pos_rate', pos_rate)
        self.log('train_mask_pos_weight_mean', avg_pos_weight)

        raise RuntimeError("Debug break")

        return total_loss

    def validation_step(self, batch: Batch, batch_idx: int):
        (
            class_loss, mask_loss, top1_acc_truth,
            has_truth_frac, num_pos_mean, ce_random_baseline,
            pos_rate, avg_pos_weight
        ) = self.forward_scm(batch)

        total_loss = class_loss + mask_loss

        self.log('val_classifier_loss', class_loss, on_epoch=True, prog_bar=True)
        self.log('val_masker_loss', mask_loss, on_epoch=True, prog_bar=True)
        self.log('val_total_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_top1_acc_truth', top1_acc_truth, on_epoch=True, prog_bar=True)
        self.log('val_has_truth_frac', has_truth_frac, on_epoch=True, prog_bar=True)
        self.log('val_num_pos_mean', num_pos_mean, on_epoch=True, prog_bar=True)
        self.log('val_ce_random_baseline', ce_random_baseline, on_epoch=True, prog_bar=True)
        self.log('val_mask_pos_rate', pos_rate, on_epoch=True, prog_bar=True)
        self.log('val_mask_pos_weight_mean', avg_pos_weight, on_epoch=True, prog_bar=True)

        return {'val_total_loss': total_loss}

    
    def on_train_epoch_start(self):
        for name, module in self.named_children():
            if name not in ['classifier', 'masker']:
                module.eval()
        self.eval()
        self.classifier.train()
        self.masker.train()



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
