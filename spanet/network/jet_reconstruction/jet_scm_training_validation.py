import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from spanet.options import Options
from spanet.network.jet_reconstruction.jet_scm_pipeline import JetSecondaryLoader
from spanet.dataset.types import Batch

tcompile = torch.compile

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return (x * weights).sum(dim=1)

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask: torch.Tensor | None = None):
        # src_key_padding_mask: (N, S) with True = ignore token
        return self.enc(x, mask=None, src_key_padding_mask=src_key_padding_mask)


class ClassifierTransformerHead(nn.Module):
    """
    Inputs:
      features_arr: (N, K, B, J, F)
      valid_mask:   (N, K) True = keep this candidate, False = duplicate to ignore
    Outputs:
      logits: (N, K)
      token_scores: (N, K)
      out_valid_mask: (N, K) mask aligned with outputs after unshuffle
    """
    def __init__(self, branch_dim: int, jets: int, feats: int,
                 class_embed_dim: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        self.branch_dim = branch_dim
        self.token_in_dim = branch_dim * jets * feats
        self.proj = nn.Linear(self.token_in_dim, class_embed_dim)
        self.tr = SimpleTransformerEncoder(class_embed_dim, nhead, num_layers, dropout)
        self.head = nn.Linear(class_embed_dim, branch_dim)
        self.norm = nn.LayerNorm(class_embed_dim)

    def forward(self, features_arr, valid_mask: torch.Tensor | None = None,
                zero_out_invalid: bool = True, neg_inf: float = -1e9):
        # features_arr: (N, K, B, J, F)
        N, K, B, J, Fdim = features_arr.shape
        device = features_arr.device

        if valid_mask is None:
            valid_mask = torch.ones((N, K), dtype=torch.bool, device=device)

        # shuffle K during training; shuffle mask identically
        if self.training:
            perms = torch.argsort(torch.rand(N, K, device=device), dim=1)
            batch_ix = torch.arange(N, device=device).unsqueeze(1)
            features_arr = features_arr[batch_ix, perms]
            valid_mask   = valid_mask[batch_ix, perms]
        else:
            perms = None

        # flatten per candidate
        tokens = features_arr.reshape(N, K, B * J * Fdim)

        # optional zeroing of masked candidates
        if zero_out_invalid:
            tokens = tokens * valid_mask.unsqueeze(-1).to(tokens.dtype)

        x = self.proj(tokens)
        x = self.norm(x)

        # mask duplicates out of attention entirely
        src_kpm = ~valid_mask  # True = ignore
        x = self.tr(x, src_key_padding_mask=src_kpm)

        per_token_branch = self.head(x)           # (N, K, B)
        token_scores = per_token_branch.max(dim=-1).values  # (N, K)
        logits = token_scores

        # unshuffle back to original order
        if perms is not None:
            inv = torch.empty_like(perms)
            inv.scatter_(1, perms, torch.arange(K, device=device).expand(N, K))
            logits       = logits[batch_ix, inv]
            token_scores = token_scores[batch_ix, inv]
            valid_mask   = valid_mask[batch_ix, inv]

        # hard-suppress invalid candidates for loss and argmax
        logits       = logits.masked_fill(~valid_mask, neg_inf)
        token_scores = token_scores.masked_fill(~valid_mask, neg_inf)

        return logits, token_scores, valid_mask


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
        self.pool = AttentionPooling(mask_embed_dim) # add attention pooling
        self.head = nn.Linear(mask_embed_dim, 1)
        self.norm = nn.LayerNorm(mask_embed_dim) # added this

    def forward(self, x):

        N, B, J, Fdim = x.shape
        # reshape to process each branch independently: batch B groups

        N, B, J, Fdim = x.shape

        x = x.reshape(N * B, J, Fdim)

        x = self.proj(x)
        x = self.norm(x) # added this

        x = self.tr(x)

        x = self.pool(x)  # Attention pooling

        logits = self.head(x).squeeze(-1).reshape(N, B)
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
        self.tr_dropout      = 0.1 # changed 0.0 -> 0.1
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

    @staticmethod
    def _dedup_valid_mask(jet_idx: torch.Tensor) -> torch.Tensor:
        """
        jet_idx: (N, K, B, J) integer indices identifying each K candidate.
        Returns is_valid: (N, K) with exactly one True per equivalence class.
        """
        N, K, B, J = jet_idx.shape
        dev = jet_idx.device
    
        # flatten and use int64 (works on CUDA)
        flat = jet_idx.reshape(N, K, B * J).to(torch.int64)
    
        FNV_OFFSET = torch.tensor(1469598103934665603, dtype=torch.int64, device=dev)
        FNV_PRIME  = torch.tensor(1099511628211,      dtype=torch.int64, device=dev)
    
        h = FNV_OFFSET.expand(N, K).clone()
        for t in range(B * J):
            h = (h ^ flat[..., t]) * FNV_PRIME
        h = h ^ (h >> 32)  # light final mix
    
        h_sorted, perm = h.sort(dim=1)
        flat_sorted = flat.gather(1, perm.unsqueeze(-1).expand_as(flat))
    
        same_hash = h_sorted[:, 1:] == h_sorted[:, :-1]
        eq_full   = same_hash & flat_sorted[:, 1:, :].eq(flat_sorted[:, :-1, :]).all(dim=-1)
    
        dup_sorted = torch.zeros((N, K), dtype=torch.bool, device=dev)
        dup_sorted[:, 1:] = eq_full  # keep first in each group
    
        inv = torch.empty_like(perm)
        inv.scatter_(1, perm, torch.arange(K, device=dev).expand(N, K))
        dup = dup_sorted.gather(1, inv)
    
        return ~dup

    @staticmethod
    def _multi_positive_ce(class_logits: torch.Tensor,
                           class_truth: torch.Tensor,
                           valid_mask: torch.Tensor | None = None):
        """
        class_logits: (N, K)
        class_truth:  (N, K) multi-hot in {0,1}
        valid_mask:   (N, K) True = include in loss
        """
        if valid_mask is None:
            valid_mask = torch.ones_like(class_truth, dtype=torch.bool, device=class_truth.device)
    
        # per-batch pos_weight from counts over NxK using only valid Ks
        with torch.no_grad():
            pos_total = (class_truth.bool() & valid_mask).sum()
            neg_total = valid_mask.sum() - pos_total
            pos_w_scalar = (neg_total.to(torch.float32) / pos_total.clamp_min(1).to(torch.float32))
            # cap to avoid extreme ratios
            cap = torch.as_tensor(1000.0, device=class_logits.device, dtype=torch.float32)
            pos_w_scalar = torch.minimum(pos_w_scalar, cap).to(class_logits.dtype)
            # same weight for all K to keep permutation equivariance
            K = class_logits.size(1)
            pos_weight = pos_w_scalar.expand(K).contiguous()
    
        weights = valid_mask.to(class_logits.dtype)
        pos_mask = class_truth.bool() & valid_mask
        has_truth = pos_mask.any(dim=1)
        num_pos = pos_mask.sum(dim=1)
    
        # avoid inf*0 on invalid Ks (classifier sets -inf there)
        safe_logits = torch.where(valid_mask, class_logits, torch.zeros_like(class_logits))
    
        bce = F.binary_cross_entropy_with_logits(
            safe_logits,
            class_truth.to(class_logits.dtype),
            reduction="none",
            pos_weight=pos_weight,
        )  # (N, K)
    
        # masked mean over classes
        denom = weights.sum(dim=1).clamp_min(1.0)
        loss_vec = (bce * weights).sum(dim=1) / denom
        loss = loss_vec[has_truth].mean() if has_truth.any() else loss_vec.mean()
    
        with torch.no_grad():
            base = F.binary_cross_entropy_with_logits(
                torch.zeros_like(class_logits),
                class_truth.to(class_logits.dtype),
                reduction="none",
                pos_weight=pos_weight,
            )
            base_vec = (base * weights).sum(dim=1) / denom
            ce_baseline = base_vec[has_truth].mean() if has_truth.any() else base_vec.mean()
    
        return loss, has_truth, num_pos, ce_baseline

    def _compiled_core(self, features_arr, pred_truth, class_truth_K_or_KBminus1, valid_mask):
        """
        valid_mask: (N, K) True=keep, False=duplicate
        """
        N, K, B, J, Fdim = features_arr.shape

        # If class_truth provided as (N, K*B-1), compress to (N, K)
        if class_truth_K_or_KBminus1.size(1) == (K * B - 1):
            ct = class_truth_K_or_KBminus1
            padded = torch.zeros(N, K * B, device=ct.device, dtype=ct.dtype)
            padded[:, :K * B - 1] = ct
            class_truth = padded.view(N, K, B).any(dim=2).to(ct.dtype)
        else:
            class_truth = class_truth_K_or_KBminus1

        # CLASSIFIER
        class_logits, token_scores, out_valid_mask = self.classifier(features_arr, valid_mask)
        class_loss, has_truth, num_pos, ce_random_baseline = \
            self._multi_positive_ce(class_logits, class_truth, valid_mask=out_valid_mask)

        rows   = torch.arange(N, device=class_logits.device)
        pred_k = token_scores.argmax(dim=1)

        # Compute top-1 vs compressed truth
        top1_acc_truth = torch.tensor(0., device=class_logits.device)
        num_pos_mean   = num_pos.float().mean()
        if has_truth.any():
            top1_acc_truth = class_truth[rows[has_truth], pred_k[has_truth]].float().mean()
            num_pos_mean   = num_pos[has_truth].float().mean()
        has_truth_frac = has_truth.float().mean()

        # MASKER
        flat_feat  = features_arr.reshape(N * K, B, J, Fdim)
        flat_truth = pred_truth.reshape(N * K, B)
        pos_rate = flat_truth.float().mean()
        logits = self.masker(flat_feat)  # (N*K, B)

        mask_loss = self.focal_bce_with_logits(
            logits, flat_truth,
            alpha_pos=self.focal_alpha_pos,
            gamma=self.focal_gamma,
            reduction="mean"
        )

        return (
            class_loss, mask_loss, top1_acc_truth,
            has_truth_frac, num_pos_mean, ce_random_baseline,
            pos_rate
        )

    # recompile after signature change
    _compiled_core = tcompile(_compiled_core, dynamic=True)

    def forward_scm(self, batch):
        pred_truth, true_masks, features_arr, class_truth, true_idx, jet_preds_tensor, jet_mult = self.topk_data(batch)

        # build per-event dedup valid mask from indices
        # jet_preds_tensor is assumed (N, K, B, J); transpose if your pipeline differs
        valid_mask = self._dedup_valid_mask(jet_preds_tensor)

        return self._compiled_core(features_arr, pred_truth, class_truth, valid_mask)

    def training_step(self, batch: Batch, batch_idx: int):
        (
            class_loss, mask_loss, top1_acc_truth,
            has_truth_frac, num_pos_mean, ce_random_baseline,
            pos_rate#, avg_pos_weight
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
        # self.log('train_mask_pos_weight_mean', avg_pos_weight)


        return total_loss

    def validation_step(self, batch: Batch, batch_idx: int):
        (
            class_loss, mask_loss, top1_acc_truth,
            has_truth_frac, num_pos_mean, ce_random_baseline,
            pos_rate#, avg_pos_weight
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
        # self.log('val_mask_pos_weight_mean', avg_pos_weight, on_epoch=True, prog_bar=True)

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
