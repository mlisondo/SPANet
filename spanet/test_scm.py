from typing import Optional, Dict

from argparse import ArgumentParser

import numpy as np
import torch                                 

from sklearn.metrics import accuracy_score, top_k_accuracy_score, ConfusionMatrixDisplay, precision_recall_curve as skl_prc
from sklearn.metrics import roc_curve as skl_roc, auc as skl_auc, confusion_matrix as skl_cm

import json, os, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from spanet.evaluation_scm import evaluate_on_test_dataset, load_model

from scipy.special import logsumexp
import hashlib

# ------------------------------------------------------------------ CLASSIFIER
def classifier_metrics(
    class_truth : np.ndarray,           # (E, K)
    class_logits: np.ndarray,           # (E, K)
    k           : int,
    valid_mask  : Optional[np.ndarray] = None    # (E,)
) -> Dict[str, float]:
    E, K = class_logits.shape
    has_truth = class_truth.any(axis=1)
    if valid_mask is None:
        valid_mask = np.ones(E, dtype=bool)

    keep = has_truth & valid_mask
    if not keep.any():
        return {"Top-1": float("nan"), "Top-1_base": float("nan"),
                **{f"Top-{m}": float("nan") for m in range(2, 2*k)},
                "has_truth_frac": 0.0}

    truth_v  = class_truth [keep]
    logits_v = class_logits[keep]

    metrics: Dict[str, float] = {}
    metrics["Top-1"] = topm_any_positive(truth_v, logits_v, 1)
    for m in range(2, min(2*k, K) + 1):
        metrics[f"Top-{m}"] = topm_any_positive(truth_v, logits_v, m)

    last_idx = K - 1
    metrics["Top-1_base"] = float((truth_v[:, last_idx] == 1).mean())
    metrics["has_truth_frac"] = float(keep.mean())
    return metrics

# helper function
def topm_any_positive(class_truth: np.ndarray, class_logits: np.ndarray, m: int) -> float:
    """
    Success if ANY true class (class_truth==1) is within the top-m scores.
    class_truth : (E, K)  multi-hot
    class_logits: (E, K)  raw logits
    """
    E, K = class_logits.shape
    m = min(m, K)
    pos = class_truth.astype(bool)
    # Get indices of the top-m scores per row (O(K) via argpartition; order inside chunk is arbitrary)
    top_idx = np.argpartition(class_logits, -m, axis=1)[:, -m:]  # not fully sorted
    # Check membership of any positive in those indices
    hit = pos[np.arange(E)[:, None], top_idx].any(axis=1)
    return float(hit.mean())

# ------------------------------------------------------------------ MASKER
def masker_metrics(mask_prob : np.ndarray,
                   mask_pred : np.ndarray,
                   mask_truth: np.ndarray) -> Dict[str, float | list]:
    """
    Precision-Recall, ROC and confusion matrix for the mask head.
    """
    P, T = mask_prob.ravel(), mask_truth.ravel()
    precision, recall, _ = skl_prc(T, P)
    fpr, tpr, _          = skl_roc(T, P)
    return {
        "AUC_PR"   : skl_auc(recall, precision),
        "AUC_ROC"  : skl_auc(fpr, tpr),
        "Confusion": skl_cm(T, mask_pred.ravel()),
        # curves returned for plotting
        "_pr_curve": (recall, precision),
        "_roc_curve": (fpr, tpr),
    }

# ------------------------------------------------------------------ JOINT
def joint_metrics(class_truth  : np.ndarray,        # (E, K)
                  class_pred   : np.ndarray,        # (E,)       secondary
                  mask_pred    : np.ndarray,        # (E, K, B)  secondary
                  mask_truth   : np.ndarray,        # (E, K, B)
                  features_arr : np.ndarray,        # (E, K, B, J, F)
                  true_masks   : np.ndarray,        # (E, K) boolean mask
                  valid_mask   : Optional[np.ndarray] = None  # override if needed
) -> Dict[str, float]:
    E, K, B = mask_pred.shape
    has_truth = class_truth.any(axis=1) # Determine events with at least one valid hypothesis
    if valid_mask is None: # Should hopefully never be none.
        valid_mask = true_masks.any(axis=-1)  # shape: (E,)

    keep = has_truth & valid_mask
    if not keep.any():
        return {k: float("nan") for k in
                ["Event_eff", "Partial_rec", "Event_eff_base", "Partial_rec_base"]}

    idx = np.where(keep)[0]
    probe(has_truth, "has_truth")
    probe(valid_mask, "valid_mask")
    print("has_truth[:10]",has_truth[:10])
    print("valid_mask[:10]",valid_mask[:10])
    probe(idx, "idx")
    print(idx)

    # ========== Truth-based eligibility ==========
    branch_counts = (mask_truth * class_truth[..., None])[idx].sum(axis=2)  # (N_keep, K)
    full_eligible = (branch_counts == B).any(axis=1)                         # (N_keep,)
    partial_eligible = ((branch_counts > 0) & (branch_counts < B)).any(axis=1)

    # ========== Secondary classifier metrics ==========
    sec_correct_cls = class_truth[idx, class_pred[idx]] == 1
    sec_matches     = (mask_pred[idx, class_pred[idx]] == mask_truth[idx, class_pred[idx]])
    n_sec_correct   = sec_matches.sum(axis=1)
    sec_full_ok     = sec_correct_cls & (n_sec_correct == B)
    sec_partial_ok  = sec_correct_cls & (n_sec_correct > 0) & (n_sec_correct < B)

    # ========== Baseline SPANet metrics ==========
    base_k = K - 1
    base_correct_cls = class_truth[idx, base_k] == 1

    # ========== Branch-level validity from true_masks ==========
    # branch_valid  = ~(features_arr[idx, base_k] == -1).any(axis=-1).any(axis=-1)        # CHANGED
    branch_valid = true_masks[idx, base_k]  # shape: (N_keep,)
    base_matches  = (branch_valid == mask_truth[idx, base_k])
    n_base_correct = base_matches.sum(axis=1)
    base_full_ok    = base_correct_cls & (n_base_correct == B)
    base_partial_ok = base_correct_cls & (n_base_correct > 0) & (n_base_correct < B)

    # ========== Aggregate metrics ==========
    n_full_elig    = full_eligible.sum()
    n_partial_elig = partial_eligible.sum()

    event_eff        = float(sec_full_ok[full_eligible].mean())   if n_full_elig else float("nan")
    partial_rec      = float(sec_partial_ok[partial_eligible].mean()) if n_partial_elig else float("nan")
    event_eff_base   = float(base_full_ok[full_eligible].mean())  if n_full_elig else float("nan")
    partial_rec_base = float(base_partial_ok[partial_eligible].mean()) if n_partial_elig else float("nan")

    return {
        "Event_eff"        : event_eff,
        "Partial_rec"      : partial_rec,
        "Event_eff_base"   : event_eff_base,
        "Partial_rec_base" : partial_rec_base,
        "_n_full_eligible"    : int(n_full_elig),
        "_n_partial_eligible" : int(n_partial_elig),
    }



# ---------------------- MAIN ----------------------


def main(
    log_directory: str,
    test_file: Optional[str],
    event_file: Optional[str],
    batch_size: Optional[int],
    gpu: bool,
    fp16: bool,             # why is this needed?
    top_k: int,
    output_dir: str):

    os.makedirs(output_dir, exist_ok=True)
    # ---------------- load model ----------------
    model = load_model(log_directory, test_file, event_file, batch_size, gpu)
    if top_k is not None:
        model.options.k = top_k

    # --------------- accumulate outputs ---------------
    arrays = evaluate_on_test_dataset(model)

    CL  = arrays["class_logits"]       # (events, K)
    CP  = arrays["class_probs"]        # (events, K)
    CPd = arrays["class_preds"]        # (events,)
    ML  = arrays["mask_logits"]        # (events, K, branches)
    MP  = arrays["mask_probs"]         # (events, K, branches)
    MPd = arrays["mask_preds"]         # (events, K, branches)
    CT  = arrays["class_truth"]        # (events, K)
    MT  = arrays["mask_truth"]         # (events, K, branches)
    feats = arrays["features_arr"]     # (events, K, branches, jets, features)
    RV = arrays["raw_valid"]           # (events,)
    TM = arrays["true_masks"]          # (events, branches)

    # ------------------ numeric + physics metrics ------------------
    # Classifier metrics
    metrics = {}
    metrics.update(classifier_metrics(CT, CL, model.options.k, valid_mask=RV))

    # Masker metrics
    m_mask = masker_metrics(MP, MPd, MT)
    metrics.update({k:v for k,v in m_mask.items() if not k.startswith("_")})

    # Joint event-level metrics
    m_joint = joint_metrics(CT, CPd, MPd, MT, feats, TM, valid_mask=RV)
    metrics.update({k:v for k,v in m_joint.items() if not k.startswith("_")})

    # curves for PDF
    recall, precision = m_mask["_pr_curve"]
    fpr, tpr          = m_mask["_roc_curve"]


    # Make numpy arrays JSON serializable
    metrics_serializable = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in metrics.items()
    }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_serializable, f, indent=4)

    # ------------------ figures ------------------
    with PdfPages(os.path.join(output_dir, "plots.pdf")) as pdf:
        # PR curve
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall")
        pdf.savefig(); plt.close()

        # ROC
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC")
        pdf.savefig(); plt.close()

        # Confusion Matrix
        plt.figure()
        cm = m_mask["Confusion"]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(cmap="Blues", values_format='d')
        plt.title("Confusion Matrix")
        pdf.savefig(); plt.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("log_directory", type=str,
                        help="Pytorch Lightning Log directory containing the checkpoint and options file.")

    parser.add_argument("-tf", "--test_file", type=str, default=None,
                        help="Replace the test file in the options with a custom one. "
                             "Must provide if options does not define a test file.")

    parser.add_argument("-ef", "--event_file", type=str, default=None,
                        help="Replace the event file in the options with a custom event.")

    parser.add_argument("-bs", "--batch_size", type=int, default=None,
                        help="Replace the batch size in the options with a custom size.")

    parser.add_argument("-g", "--gpu", action="store_true",
                        help="Evaluate network on the gpu.")
    
    parser.add_argument("-k", "--top_k", type=int, default=None,
                        help="k value override in top-k inference")
    
    parser.add_argument("-o", "--output_dir", type=str, required=True, 
                        help="Directory where metrics.json and plots.pdf will be written.")


    parser.add_argument("-fp16", "--fp16", action="store_true", help="Use Torch AMP for training.")

    arguments = parser.parse_args()
    main(**arguments.__dict__)




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