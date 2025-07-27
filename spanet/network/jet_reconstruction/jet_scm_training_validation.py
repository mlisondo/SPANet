import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List, Dict
from spanet.options import Options
from spanet.network.jet_reconstruction.jet_scm_pipeline import JetSecondaryLoader
from spanet.dataset.types import Batch

tcompile = torch.compile

class SCM_Training_Val(JetSecondaryLoader):
    def __init__(self, options: Options, torch_script: bool = False):
        '''
        input_dim and hidden_dims are defined in options file
        '''
        super(SCM_Training_Val, self).__init__(options, torch_script)
        self.options = options
        real_K = self.options.branch_dim * self.options.k - 1

        # -- Classifier Head (whole event) --
        # For each event, flatten all hypothesis/branch/jet/feature into a single vector
        class_input_dim = self.options.branch_dim * self.options.jet_max_dim * self.options.features_dim * real_K
        classifier_layers = []
        class_dims = [class_input_dim] + options.class_hidden_dims + [real_K]
        for in_d, out_d in zip(class_dims[:-1], class_dims[1:]):
            classifier_layers.append(nn.Linear(in_d, out_d))
            if out_d != real_K:
                classifier_layers.append(nn.ReLU())
        self.classifier = nn.Sequential(*classifier_layers)

        # -- Masker Head (branch-level) --
        # Masker processes each branch in each hypothesis individually
        masker_input_dim = self.options.branch_dim * self.options.jet_max_dim * self.options.features_dim
        masker_layers = []
        mask_dims = [masker_input_dim] + options.mask_hidden_dims + [self.options.branch_dim]
        for in_d, out_d in zip(mask_dims[:-1], mask_dims[1:]):
            masker_layers.append(nn.Linear(in_d, out_d))
            if out_d != self.options.branch_dim:
                masker_layers.append(nn.ReLU())
        self.masker = nn.Sequential(*masker_layers)


        self.classifier = tcompile(self.classifier, dynamic=True)
        self.masker     = tcompile(self.masker,    dynamic=True)


        # Configurable caps/defaults for imbalance handling
        self.pos_weight_cap = 1000.0
        self.use_focal_masker = "use_focal_masker"
        self.focal_alpha_pos = 0.7
        self.focal_gamma = 2.0


    def _compiled_core(self, features_arr, pred_truth, class_truth):
        events, K, branches, jets, feats = features_arr.shape
        class_in = features_arr.reshape(events, -1)  # [N, K*B*J*F]
        expected_in = K * branches * jets * feats

        class_logits = self.classifier(class_in)     # [N, real_K]

        class_loss, has_truth, num_pos, ce_random_baseline = self._multi_positive_ce(class_logits, class_truth)

        rows = torch.arange(events, device=class_logits.device)
        pred_k = torch.argmax(class_logits, dim=1)
        if has_truth.any():
            top1_acc_truth = class_truth[rows[has_truth], pred_k[has_truth]].float().mean()
            num_pos_mean = num_pos[has_truth].float().mean()
        else:
            top1_acc_truth = torch.tensor(0.0, device=class_logits.device)
            num_pos_mean = num_pos.float().mean()
        has_truth_frac = has_truth.float().mean()

        # Masker
        flat = features_arr.reshape(events * K, branches * jets * feats)
        logits_all = self.masker(flat).view(events, K, branches)  # [N, K, B]

        t = pred_truth.float()  # ensure {0,1}
        # Per-branch prevalence
        pos_per_branch = t.sum(dim=(0, 1))  # [B]
        tot_per_branch = torch.tensor(events * K, device=t.device, dtype=t.dtype)
        neg_per_branch = tot_per_branch - pos_per_branch

        eps = torch.finfo(t.dtype).eps
        pos_weight_b = (neg_per_branch / (pos_per_branch + eps)).clamp(max=self.pos_weight_cap)  # [B]

        mask_loss = self.focal_bce_with_logits(
            logits_all, t,
            alpha_pos=self.focal_alpha_pos,
            gamma=self.focal_gamma,
            reduction="mean",
        )

        pos_rate = (pos_per_branch.sum() / (events * K * branches)).to(logits_all.dtype)
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

    # single call covers whole tensor graph
    _compiled_core = tcompile(_compiled_core, dynamic=True)

    def forward_scm(self, batch):

        pred_truth, true_masks, features_arr, class_truth, true_idx, jet_preds_tensor = self.topk_data(batch)

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
