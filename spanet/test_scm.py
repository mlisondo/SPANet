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

def classifier_metrics(class_truth: np.ndarray,
                       class_logits: np.ndarray,
                       class_pred: np.ndarray,
                       features_arr: np.ndarray,
                       k: int) -> dict[str, float]:
    """
    Group duplicates per event using features, aggregate logits with log-sum-exp,
    and compute Top-m(any-positive) in group space.
    Also reports Top-1_chosen: whether your chosen index lies in a positive group.
    """
    E, K = class_logits.shape
    metrics = {}

    # Build groups per event once
    groups_per_event = [_group_event_by_features(features_arr[e]) for e in range(E)]

    # Top-1 (grouped, logits-based)
    hits_top1 = []
    for e in range(E):
        hit = _topm_any_positive_grouped(class_truth[e], class_logits[e], groups_per_event[e], m=1)
        hits_top1.append(hit)
    metrics["Top-1"] = float(np.mean(hits_top1))

    # Top-m
    max_m = min(2 * k, max(len(g) for g in groups_per_event))
    for m in range(2, max_m + 1):
        hits = []
        for e in range(E):
            hits.append(_topm_any_positive_grouped(class_truth[e], class_logits[e], groups_per_event[e], m))
        metrics[f"Top-{m}"] = float(np.mean(hits))

    # Optional diagnostic: was the chosen class inside any positive group?
    pos = class_truth.astype(bool)
    chosen_positive = pos[np.arange(E), class_pred]
    metrics["Top-1_chosen"] = float(chosen_positive.mean())

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
    Event reconstruction efficiency (all branches right) and
    partial reconstruction rate (at least one branch right).
    """
    E, K, B       = mask_pred.shape
    correct_class = class_truth[np.arange(E), class_pred] == 1
    matches       = (mask_pred[np.arange(E), class_pred] == mask_truth[np.arange(E), class_pred])
    n_correct     = matches.sum(axis=1)

    full_mask   = (n_correct == B)
    partial_mask= (n_correct > 0) & (n_correct < B)

    return {
        "Event_eff"  : float(np.mean(correct_class & full_mask)),
        "Partial_rec": float(np.mean(correct_class & partial_mask)),
        "_full_mask" : full_mask,   # keep for later physics slicing
        "_partial_mask": partial_mask
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
    metrics.update(classifier_metrics(CT, CL, CPd, model.options.k))

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
