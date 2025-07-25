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

    def _compiled_core(self, features_arr, pred_truth, class_truth):
        print("[DEBUG] --> Inside _compiled_core")

        e = 0

        """Tensor-only slice of forward_scm."""
        events, K, branches, jets, feats = features_arr.shape
        class_in = features_arr.reshape(events, -1)

        print("\n[DEBUG] Classifier Input [0]:")
        print(class_in[e])  # Shape: (flat_dim,)

        class_logits = self.classifier(class_in)

        print("[DEBUG] Classifier Output Logits [0]:")
        print(class_logits[e])  # Shape: (K,)


        class_truth_int = class_truth.to(torch.int)
        class_first = torch.argmax(class_truth_int, 1)
        has_truth   = torch.any(class_truth_int == 1, 1)

        print("[DEBUG] Classifier Truth (argmax) [0]:", class_first[e].item())
        print("[DEBUG] Classifier Truth (raw):", class_truth[e])

        mask = ~class_truth.bool().clone()
        rows = torch.arange(events, device=class_logits.device)
        
        mask[rows[has_truth], class_first[has_truth]] = False
        neg_inf = torch.finfo(class_logits.dtype).min
        masked_logits = class_logits.masked_fill(mask, neg_inf)

        print("[DEBUG] Masked Classifier Logits [0]:")
        print(masked_logits[e])
        
        class_loss = nn.CrossEntropyLoss(reduction="none")(
            class_logits, class_first)[has_truth].mean()
        print(f"[DEBUG] Classifier Loss: {class_loss.item():.4f}")

        # vectorised masker
        flat = features_arr.reshape(events*K, branches*jets*feats)
        print("\n[DEBUG] Masker Input [0]:")
        print(flat[e * K])

        logits_all = self.masker(flat).view(events, K, branches)
        print("[DEBUG] Masker Logits [0]:")
        print(logits_all[e])  # Shape: (K, B)

        mask_loss  = nn.BCEWithLogitsLoss()(logits_all,
                                            pred_truth.float())
        print(f"[DEBUG] Masker Loss: {mask_loss.item():.4f}")
        
        pred_k = torch.argmax(class_logits, 1)
        print(f"[DEBUG] pred_k: {pred_k}")
        top1_acc = class_truth[rows, pred_k].float().mean()
        print(f"[DEBUG] Top 1 Acc: {top1_acc}")

        return class_loss, mask_loss, top1_acc
    
    # single call covers whole tensor graph
    _compiled_core = tcompile(_compiled_core, dynamic=True)

    def forward_scm(self, batch):
        print("[DEBUG] --> Entered forward_scm")

        pred_truth, true_masks, features_arr, class_truth = self.topk_data(batch)

        e = 0

        print("\n======== FORWARD: Event 0 Inputs ========")
        print("pred_truth[0]:")
        print(pred_truth[e])  # (K, B)

        print("true_masks[:, 0]:")
        print(true_masks[:, e])  # (B,)

        print("class_truth[0]:")
        print(class_truth[e])  # (K,)

        print("features_arr[0]:")
        print(features_arr[e])  # (K, B, J, F)

        print("=" * 40)


        return self._compiled_core(features_arr, pred_truth, class_truth)


    def training_step(self, batch: Batch, batch_idx: int) -> Dict[str, torch.Tensor]:

        self.on_train_epoch_start()

        class_loss, mask_loss, top1_acc = self.forward_scm(batch)

        total_loss = class_loss + mask_loss

        self.log('train_classifier_loss', class_loss)
        self.log('train_masker_loss', mask_loss)
        self.log('train_total_loss', total_loss)
        self.log('train_top1_acc', top1_acc)


        raise RuntimeError("Debug break")

        return total_loss
        
    def validation_step(self, batch: Batch, batch_idx: int) -> Dict[str, torch.Tensor]:

        class_loss, mask_loss, top1_acc = self.forward_scm(batch)
        total_loss = class_loss + mask_loss

        self.log('val_classifier_loss', class_loss, on_epoch=True, prog_bar=True)
        self.log('val_masker_loss', mask_loss, on_epoch=True, prog_bar=True)
        self.log('val_total_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_top1_acc', top1_acc, on_epoch=True, prog_bar=True)

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
