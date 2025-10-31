from typing import Optional, Dict

from argparse import ArgumentParser

import numpy as np
import torch                                 

from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve as skl_prc
from sklearn.metrics import roc_curve as skl_roc, auc as skl_auc, confusion_matrix as skl_cm

import json, os, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from spanet.evaluation_scm import evaluate_on_test_dataset, load_model

import hashlib

# ------------------------------------------------------------------ CLASSIFIER
def classifier_metrics(
    class_truth : np.ndarray,           # (E, K)
    class_logits: np.ndarray,           # (E, K)
    k           : int,
    valid_mask  : Optional[np.ndarray] = None      # (E,)
    ) -> Dict[str, float]:
    E, K = class_logits.shape
    if valid_mask is None:
        valid_mask = np.ones(E, dtype=bool)

    # Eligibility mask: events that actually contain at least 1 positive label
    has_truth = class_truth.any(axis=1)
    keep      = has_truth & valid_mask
    if not keep.any():
        nan = float("nan")
        return {
            **{f"Top-{m}"          : nan for m in range(1, min(2 * k, K) + 1)},
            **{f"Top-{m}_strict"   : nan for m in range(1, min(2 * k, K) + 1)},
            "Top-1_base"           : nan,
            "Top-1_base_strict"    : nan,
            "has_truth_frac"       : 0.0,
        }

    truth_v  = class_truth[keep].astype(bool)   # (N, K)
    logits_v = class_logits[keep]              # (N, K)
    N        = truth_v.shape[0]

    # Sort logits once -> easy top-m slices
    sorted_idx = np.argsort(-logits_v, axis=1)  # (N, K) descending

    # Lenient
    metrics: Dict[str, float] = {}
    for m in range(1, min(2 * k, K) + 1):
        topm_idx = sorted_idx[:, :m]            # (N, m)

        # at least one positive label among top-m
        hit_any  = truth_v[np.arange(N)[:, None], topm_idx].any(axis=1)
        key      = "Top-1" if m == 1 else f"Top-{m}"
        metrics[key]      = float(hit_any.mean())

    # Baseline SPANet hypothesis (index K-1)
    base_is_pos = truth_v[:, K - 1]
    metrics["Top-1_base"] = float(base_is_pos.mean())

    metrics["has_truth_frac"] = float(keep.mean()) # extra
    return metrics

# ------------------------------------------------------------------ MASKER
def masker_metrics(mask_prob : np.ndarray,
                   mask_pred : np.ndarray,
                   pred_truth: np.ndarray) -> Dict[str, float | list]:
    """
    Precision-Recall, ROC and confusion matrix for the mask head.
    """
    P, T = mask_prob.ravel(), pred_truth.ravel()
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
                  class_pred   : np.ndarray,        # (E,) classifier hypothesis index
                  mask_pred    : np.ndarray,        # (E, K, B)
                  pred_truth   : np.ndarray,        # (E, K, B)
                  true_masks   : np.ndarray         # (E, B)
    ) -> Dict[str, float]:
    E, K, B = mask_pred.shape

    # Reconstruction categorization
    branch_valid_count = true_masks.sum(axis=1)
    is_full_reco       = (branch_valid_count == B)
    is_partial_reco    = (branch_valid_count > 0) & (branch_valid_count < B)
    n_full_reco    = is_full_reco.sum()
    n_partial_reco = is_partial_reco.sum()
    
    # METRICS: every branch (valid or not) must match the ground-truth mask
    correct_hypothesis = class_truth[np.arange(E), class_pred] == 1
    mask_chosen  = mask_pred[np.arange(E), class_pred]   # (E, B)
    correct_mask = np.all(mask_chosen == true_masks, axis=1)
    correct_base_mask = np.all(mask_pred[:, -1] == pred_truth[:, -1], axis=1)
    correct_both = correct_hypothesis & correct_mask

    correct_hypothesis_base = class_truth[np.arange(E), -1] == 1
    correct_both_base = correct_hypothesis_base & correct_base_mask
    truth_available = np.any(class_truth, axis=1)

    event_eff        = correct_both[is_full_reco].mean()         if n_full_reco else float('nan')
    partial_eff      = correct_both[is_partial_reco].mean()      if n_partial_reco else float('nan')
    event_eff_no_mask   = correct_hypothesis[is_full_reco].mean()    if n_full_reco else float('nan')
    partial_eff_no_mask = correct_hypothesis[is_partial_reco].mean() if n_partial_reco else float('nan')

    event_eff_base        = correct_both_base[is_full_reco].mean()         if n_full_reco else float('nan')
    partial_eff_base      = correct_both_base[is_partial_reco].mean()      if n_partial_reco else float('nan')
    event_eff_no_mask_base   = correct_hypothesis_base[is_full_reco].mean()    if n_full_reco else float('nan')
    partial_eff_no_mask_base = correct_hypothesis_base[is_partial_reco].mean() if n_partial_reco else float('nan')
    event_eff_no_mask_base_max   = truth_available[is_full_reco].mean()    if n_full_reco else float('nan')
    partial_eff_no_mask_base_max = truth_available[is_partial_reco].mean() if n_partial_reco else float('nan')

    return {
        "event_eff"        : float(event_eff),
        "partial_eff"      : float(partial_eff),
        "event_eff_no_mask"   : float(event_eff_no_mask),
        "partial_eff_no_mask" : float(partial_eff_no_mask),
        "event_eff_base"         : float(event_eff_base),
        "partial_eff_base"       : float(partial_eff_base),
        "event_eff_no_mask_base"    : float(event_eff_no_mask_base),
        "partial_eff_no_mask_base"  : float(partial_eff_no_mask_base),
        "event_eff_no_mask_base_max"    : float(event_eff_no_mask_base_max),
        "partial_eff_no_mask_base_max"  : float(partial_eff_no_mask_base_max),
        "_n_full_eligible"    : int(n_full_reco),
        "_n_partial_eligible" : int(n_partial_reco),
    }

# ------------------------------------------------------------------ REAL
def strict_metric(class_truth  : np.ndarray,        # (E, K)
                  class_pred   : np.ndarray,        # (E,)   SCM‑chosen hypothesis index
                  mask_pred    : np.ndarray,        # (E, K, B)
                  pred_truth   : np.ndarray,        # (E, K, B)  branch‑level truth match flags
                  true_masks   : np.ndarray         # (E, B)     “reconstructable” mask
    ) -> Dict[str, float]:
    """
    real_event_eff         - SCM:  all branches AND mask exactly match ground truth
    real_event_eff_base    - SPANet baseline (index K-1) under the same strict rule
    """
    E, K, B = mask_pred.shape
    base_k  = K - 1
    # strict correctness for SCM-chosen hypothesis
    scm_all_branches_ok = np.all(pred_truth[np.arange(E), class_pred] == true_masks, axis=1)
    scm_mask_ok         = np.all(mask_pred[np.arange(E), class_pred]  == true_masks, axis=1)
    strict_scm_correct  = scm_all_branches_ok & scm_mask_ok         # (E,)

    # 2.  Strict correctness for SPANet baseline (hypothesis K-1) ------------
    base_all_branches_ok = np.all(pred_truth[:, base_k] == true_masks, axis=1)
    base_mask_ok         = np.all(mask_pred[:, base_k]  == true_masks, axis=1)
    strict_base_correct  = base_all_branches_ok & base_mask_ok        # (E,)

    real_event_eff       = strict_scm_correct.mean()   if E else float("nan")
    real_event_eff_base  = strict_base_correct.mean()  if E else float("nan")

    return {
        "real_event_eff"       : float(real_event_eff),
        "real_event_eff_base"  : float(real_event_eff_base),
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
    PT  = arrays["pred_truth"]         # (events, K, branches)
    feats = arrays["features_arr"]     # (events, K, branches, jets, features)
    RV = arrays["raw_valid"]           # (events,)
    TM = arrays["true_masks"]          # (events, branches)
    JM = arrays["jet_mult"]            # (events, Njets)        (for ttbar (Njets) = 10, true if jet is valid)

    # inclusive
    ICL  = arrays["inclusive_class_logits"]       # (events, K)
    ICP  = arrays["inclusive_class_probs"]        # (events, K)
    ICPd = arrays["inclusive_class_preds"]        # (events,)
    IML  = arrays["inclusive_mask_logits"]        # (events, K, branches)
    IMP  = arrays["inclusive_mask_probs"]         # (events, K, branches)
    IMPd = arrays["inclusive_mask_preds"]         # (events, K, branches)
    # prior
    PCL  = arrays["prior_class_logits"]       # (events, K)
    PCP  = arrays["prior_class_probs"]        # (events, K)
    PCPd = arrays["prior_class_preds"]        # (events,)
    PML  = arrays["prior_mask_logits"]        # (events, K, branches)
    PMP  = arrays["prior_mask_probs"]         # (events, K, branches)
    PMPd = arrays["prior_mask_preds"]         # (events, K, branches)


    # ------------------ MULTIPLICITY METRICS ------------------
    metrics = {}
    n_jets = JM.sum(axis=1).astype(np.int64)     # (E,)

    for i in (6, 7, 8, 1):
        if i == 8:
            chosen = (n_jets >= 8)               # 8+
            tag    = "njets_8p"
        elif i == 1:
            chosen = np.ones(n_jets.shape[0], dtype=bool)  # inclusive
            tag    = "inclusive"
        else:
            chosen = (n_jets == i)
            tag    = f"njets_{i}"

        count = int(chosen.sum())
        metrics[f"{tag}/count"] = count
        if count == 0:
            continue

        _CL  = CL[chosen];  _CPd = CPd[chosen]
        _MP  = MP[chosen];  _MPd = MPd[chosen]
        _CT  = CT[chosen];  _PT  = PT[chosen]
        _TM  = TM[chosen];  _RV  = RV[chosen]

        # inclusive
        _ICL  = ICL[chosen];  _ICPd = ICPd[chosen]
        _IMP  = IMP[chosen];  _IMPd = IMPd[chosen]
        # prior 
        _PCL  = PCL[chosen];  _PCPd = PCPd[chosen]
        _PMP  = PMP[chosen];  _PMPd = PMPd[chosen]

        # --- Classifier ---
        m_cls = classifier_metrics(_CT, _CL, model.options.k, valid_mask=_RV)
        metrics.update({f"{tag}/{k}": v for k, v in m_cls.items()})
        m_cls = classifier_metrics(_CT, _ICL, model.options.k, valid_mask=_RV)
        metrics.update({f"INCLUSIVE/{tag}/{k}": v for k, v in m_cls.items()})
        m_cls = classifier_metrics(_CT, _PCL, model.options.k, valid_mask=_RV)
        metrics.update({f"PRIOR/{tag}/{k}": v for k, v in m_cls.items()})

        # --- Masker ---
        m_mask = masker_metrics(_MP, _MPd, _PT)
        metrics.update({f"{tag}/{k}": v for k, v in m_mask.items() if not k.startswith("_")})
        m_mask = masker_metrics(_IMP, _IMPd, _PT)
        metrics.update({f"INCLUSIVE/{tag}/{k}": v for k, v in m_mask.items() if not k.startswith("_")})
        m_mask = masker_metrics(_PMP, _PMPd, _PT)
        metrics.update({f"PRIOR/{tag}/{k}": v for k, v in m_mask.items() if not k.startswith("_")})

        # --- Joint ---
        m_joint = joint_metrics(_CT, _CPd, _MPd, _PT, _TM)
        metrics.update({f"{tag}/{k}": v for k, v in m_joint.items() if not k.startswith("_")})
        m_joint = joint_metrics(_CT, _ICPd, _IMPd, _PT, _TM)
        metrics.update({f"INCLUSIVE/{tag}/{k}": v for k, v in m_joint.items() if not k.startswith("_")})
        m_joint = joint_metrics(_CT, _PCPd, _PMPd, _PT, _TM)
        metrics.update({f"PRIOR/{tag}/{k}": v for k, v in m_joint.items() if not k.startswith("_")})

        # --- Strict ---
        m_real = strict_metric(_CT, _CPd, _MPd, _PT, _TM)
        metrics.update({f"{tag}/{k}": v for k, v in m_real.items() if not k.startswith("_")})
        m_real = strict_metric(_CT, _ICPd, _IMPd, _PT, _TM)
        metrics.update({f"INCLUSIVE/{tag}/{k}": v for k, v in m_real.items() if not k.startswith("_")})
        m_real = strict_metric(_CT, _PCPd, _PMPd, _PT, _TM)
        metrics.update({f"PRIOR/{tag}/{k}": v for k, v in m_real.items() if not k.startswith("_")})

    # # ------------------ HISTOGRAMS ------------------

    # Make numpy arrays JSON serializable
    metrics_serializable = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in metrics.items()
    }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_serializable, f, indent=4)


    # ALL FIGURES CAN BE MADE FROM METRICS (except for MASKER specific curves, which i do not care for right now)


    # # curves for PDF
    # recall, precision = m_mask["_pr_curve"]
    # fpr, tpr = m_mask["_roc_curve"]

    # # SCM vs Baseline comparison
    # scm_keys  = ["event_eff", "event_eff_no_mask", "partial_eff", "partial_eff_no_mask"]
    # base_keys = [k + "_base" for k in scm_keys]
    # scm_vals  = [metrics_serializable[k] for k in scm_keys]
    # base_vals = [metrics_serializable[k] for k in base_keys]

    # # ------------------ figures ------------------
    # with PdfPages(os.path.join(output_dir, "plots.pdf")) as pdf:
    #     # PR curve
    #     plt.figure()
    #     plt.plot(recall, precision)
    #     plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall")
    #     pdf.savefig(); plt.close()

    #     # ROC
    #     plt.figure()
    #     plt.plot(fpr, tpr)
    #     plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    #     plt.title("ROC")
    #     pdf.savefig(); plt.close()

    #     # Confusion Matrix
    #     plt.figure()
    #     cm = m_mask["Confusion"]
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    #     disp.plot(cmap="Blues", values_format='d')
    #     plt.title("Confusion Matrix")
    #     pdf.savefig(); plt.close()

    #     # Comparison Histogram
    #     x = np.arange(len(scm_keys))
    #     w = 0.35
    #     plt.figure(figsize=(8, 4))
    #     plt.bar(x - w/2, scm_vals, width=w, label="SCM")
    #     plt.bar(x + w/2, base_vals, width=w, label="Baseline")
    #     plt.xticks(x, ["Full Reco", "Full Reco\nno mask", "Partial Reco", "Partial Reco\nno mask"])
    #     plt.ylabel("Accuracy")
    #     plt.ylim(0, 1.0)
    #     plt.title("SCM Extension vs SPANet Baseline")
    #     plt.legend(loc="best")
    #     pdf.savefig(); plt.close()

    #     # Real-event efficiency ("strict") histogram
    #     plt.figure(figsize=(4, 4))
    #     strict_vals = [
    #         metrics_serializable["real_event_eff"],
    #         metrics_serializable["real_event_eff_base"],
    #     ]
    #     plt.bar(["SCM", "Baseline"], strict_vals, width=0.5)
    #     plt.ylabel("Strict reconstruction efficiency")
    #     plt.ylim(0, 1.0)
    #     plt.title("Real Event Efficiency (strict)")
    #     for i, v in enumerate(strict_vals):
    #         plt.text(i, v + 0.02, f"{v:.3f}",
    #                  ha="center", va="bottom", fontsize=8)
    #     pdf.savefig(); plt.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--log_directory", type=str,
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
