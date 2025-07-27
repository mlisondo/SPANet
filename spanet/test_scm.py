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

def _group_event_by_features(features_e: np.ndarray, precision: int = 6):
    """
    Group duplicate hypotheses for one event by exact feature identity (within rounding).
    features_e: (K, B, J, F)
    Returns:
        groups: list[np.ndarray] each with member k indices
    """
    K = features_e.shape[0]
    flat = np.round(features_e.reshape(K, -1), precision)
    keys = [hashlib.sha1(row.tobytes()).hexdigest() for row in flat]
    key_to_members = {}
    for k, key in enumerate(keys):
        key_to_members.setdefault(key, []).append(k)
    return [np.asarray(m, dtype=np.int32) for m in key_to_members.values()]

def _topm_any_positive_grouped(class_truth_e: np.ndarray,
                               class_logits_e: np.ndarray,
                               groups: list[np.ndarray],
                               m: int) -> bool:
    """
    One event: success if ANY positive group is within top-m by LSE-aggregated logits.
    """
    G = len(groups)
    g_logits = np.empty(G, dtype=class_logits_e.dtype)
    g_truth  = np.empty(G, dtype=bool)
    for g, members in enumerate(groups):
        g_logits[g] = logsumexp(class_logits_e[members])
        g_truth[g]  = class_truth_e[members].astype(bool).any()

    m = min(m, G)
    top_idx = np.argpartition(g_logits, -m)[-m:]
    return bool(g_truth[top_idx].any())

# ------------------------------------------------------------------ CLASSIFIER
def classifier_metrics(
    class_truth : np.ndarray,        # (E, K)   multi-hot
    class_logits: np.ndarray,        # (E, K)   raw logits
    class_pred : np.ndarray,         # (E,)     “chosen” index (still logged)
    k           : int
) -> Dict[str, float]:
    """
    Top-m(any-positive) *only on events that have >=1 positive class.
    Adds two diagnostics:
      • has_truth_frac  – fraction of events that were evaluated
      • Top-1_chosen    – hit-rate of the user-supplied class_pred
    """
    E, K = class_logits.shape
    has_truth = class_truth.any(axis=1)            # (E,)

    if not has_truth.any():                        # degenerate edge-case
        return {f"Top-{m}": float('nan') for m in range(1, 2*k)} | {
                "has_truth_frac": 0.0, "Top-1_chosen": float('nan')}

    truth_valid  = class_truth [has_truth]
    logits_valid = class_logits[has_truth]

    metrics: Dict[str, float] = {}
    metrics["Top-1"] = topm_any_positive(truth_valid, logits_valid, 1)

    for m in range(2, min(2*k, K) + 1):
        metrics[f"Top-{m}"] = topm_any_positive(truth_valid, logits_valid, m)

    # Diagnostics
    metrics["has_truth_frac"] = float(has_truth.mean())
    metrics["Top-1_chosen"]   = float(
        class_truth[has_truth].astype(bool)
        [np.arange(has_truth.sum()), class_pred[has_truth]].mean()
    )
    return metrics


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

# --------------------------------------------------------- EVENT-LEVEL JOINT
def joint_metrics(class_truth : np.ndarray,
                  class_pred  : np.ndarray,
                  mask_pred   : np.ndarray,
                  mask_truth  : np.ndarray) -> Dict[str, float]:
    """
    Event efficiency and partial-reconstruction evaluated only where a
    positive class exists.
    """
    E, K, B = mask_pred.shape
    has_truth = class_truth.any(axis=1)            # (E,)
    if not has_truth.any():
        return {"Event_eff": float('nan'), "Partial_rec": float('nan')}

    idx            = np.where(has_truth)[0]        # indices to keep
    correct_class  = class_truth[idx, class_pred[idx]] == 1
    matches        = (mask_pred[idx, class_pred[idx]] ==
                      mask_truth[idx, class_pred[idx]])
    n_correct      = matches.sum(axis=1)

    full_mask      = (n_correct == B)
    partial_mask   = (n_correct > 0) & (n_correct < B)

    return {
        "Event_eff"  : float(np.mean(correct_class & full_mask)),
        "Partial_rec": float(np.mean(correct_class & partial_mask)),
        # keep masks for later slicing
        "_full_mask"   : full_mask,
        "_partial_mask": partial_mask,
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

    # ------------------ numeric + physics metrics ------------------
    metrics = {}
    metrics.update(classifier_metrics(CT, CL, CPd, feats, model.options.k))

    m_mask = masker_metrics(MP, MPd, MT)
    metrics.update({k:v for k,v in m_mask.items() if not k.startswith("_")})

    m_joint = joint_metrics(CT, CPd, MPd, MT)
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
