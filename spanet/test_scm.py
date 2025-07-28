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
# def classifier_metrics(
#     class_truth : np.ndarray,        # (E, K)
#     class_logits: np.ndarray,        # (E, K)
#     class_pred : np.ndarray,         # (E,)
#     k           : int,
#     features_arr: np.ndarray         # (E, K, B, J, F)
# ) -> Dict[str, float]:
#     """
#     Top-m(any-positive) evaluated using hypothesis groupings, for events with >=1 positive class.
#     Adds:
#       • has_truth_frac  – fraction of events with class_truth
#       • Top-1_chosen    – was chosen hypothesis positive
#     """
#     E, K = class_logits.shape
#     has_truth = class_truth.any(axis=1)            # (E,)

#     if not has_truth.any():
#         return {f"Top-{m}": float('nan') for m in range(1, 2*k)} | {
#                 "has_truth_frac": 0.0, "Top-1_chosen": float('nan')}

#     truth_valid  = class_truth [has_truth]         # (E_valid, K)
#     logits_valid = class_logits[has_truth]         # (E_valid, K)
#     feats_valid  = features_arr[has_truth]         # (E_valid, K, B, J, F)

#     metrics: Dict[str, float] = {}

#     for m in range(1, min(2*k, K) + 1):
#         topm_hits = 0
#         for i in range(truth_valid.shape[0]):
#             features_e     = feats_valid[i]         # (K, B, J, F)
#             class_truth_e  = truth_valid[i]         # (K,)
#             class_logits_e = logits_valid[i]        # (K,)
#             groups         = _group_event_by_features(features_e)
#             topm_hits     += _topm_any_positive_grouped(class_truth_e, class_logits_e, groups, m)
#         metrics[f"Top-{m}"] = topm_hits / truth_valid.shape[0]
        
#     metrics["has_truth_frac"] = float(has_truth.mean())
#     metrics["Top-1_chosen"] = float(
#         class_truth[has_truth].astype(bool)
#         [np.arange(has_truth.sum()), class_pred[has_truth]].mean()
#     )
#     return metrics
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
# def joint_metrics(class_truth : np.ndarray,
#                   class_pred  : np.ndarray,
#                   mask_pred   : np.ndarray,
#                   mask_truth  : np.ndarray) -> Dict[str, float]:
#     """
#     Event efficiency and partial-reconstruction evaluated only where a
#     positive class exists.
#     """
#     E, K, B = mask_pred.shape
#     has_truth = class_truth.any(axis=1)            # (E,)
#     if not has_truth.any():
#         return {"Event_eff": float('nan'), "Partial_rec": float('nan')}

#     idx            = np.where(has_truth)[0]        # indices to keep
#     correct_class  = class_truth[idx, class_pred[idx]] == 1
#     matches        = (mask_pred[idx, class_pred[idx]] ==
#                       mask_truth[idx, class_pred[idx]])
#     n_correct      = matches.sum(axis=1)

#     full_mask      = (n_correct == B)
#     partial_mask   = (n_correct > 0) & (n_correct < B)

#     return {
#         "Event_eff"  : float(np.mean(correct_class & full_mask)),
#         "Partial_rec": float(np.mean(correct_class & partial_mask)),
#         # keep masks for later slicing
#         "_full_mask"   : full_mask,
#         "_partial_mask": partial_mask,
#     }
def joint_metrics(class_truth  : np.ndarray,        # (E, K)
                  class_pred   : np.ndarray,        # (E,)
                  mask_pred    : np.ndarray,        # (E, K, B)
                  mask_truth   : np.ndarray,        # (E, K, B)
                  features_arr : np.ndarray,        # (E, K, B, J, F)
                  valid_mask   : Optional[np.ndarray] = None
) -> Dict[str, float]:
    E, K, B = mask_pred.shape
    has_truth = class_truth.any(axis=1)
    if valid_mask is None:
        valid_mask = np.ones(E, dtype=bool)

    keep = has_truth & valid_mask
    if not keep.any():
        return {"Event_eff": float("nan"),       "Partial_rec": float("nan"),
                "Event_eff_base": float("nan"), "Partial_rec_base": float("nan")}

    idx = np.where(keep)[0]

    # Secondary model
    sec_correct_cls = class_truth[idx, class_pred[idx]] == 1
    sec_matches     = (mask_pred[idx, class_pred[idx]] ==
                       mask_truth[idx, class_pred[idx]])
    n_sec_correct   = sec_matches.sum(axis=1)
    sec_full    = (n_sec_correct == B)
    sec_partial = (n_sec_correct > 0) & (n_sec_correct < B)

    # Baseline SPANet (last hypothesis)
    base_k        = K - 1
    branch_valid  = ~(features_arr[idx, base_k] == -1).any(axis=-1).any(axis=-1)
    base_correct_cls = class_truth[idx, base_k] == 1
    base_matches     = (branch_valid == mask_truth[idx, base_k])
    n_base_correct   = base_matches.sum(axis=1)
    base_full    = (n_base_correct == B)
    base_partial = (n_base_correct > 0) & (n_base_correct < B)

    return {
        "Event_eff"         : float(np.mean(sec_correct_cls & sec_full)),
        "Partial_rec"       : float(np.mean(sec_correct_cls & sec_partial)),
        "Event_eff_base"    : float(np.mean(base_correct_cls & base_full)),
        "Partial_rec_base"  : float(np.mean(base_correct_cls & base_partial)),
        "_full_mask"        : sec_full,
        "_partial_mask"     : sec_partial,
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
    RV = arrays["raw_valid"]  

    # ------------------ numeric + physics metrics ------------------
    # Classifier metrics
    metrics = {}
    metrics.update(classifier_metrics(CT, CL, model.options.k, valid_mask=RV))

    # Masker metrics
    m_mask = masker_metrics(MP, MPd, MT)
    metrics.update({k:v for k,v in m_mask.items() if not k.startswith("_")})

    # Joint event-level metrics
    m_joint = joint_metrics(CT, CPd, MPd, MT, feats, valid_mask=RV)
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
