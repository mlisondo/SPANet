from typing import Optional, Dict

from argparse import ArgumentParser

import numpy as np
import torch                                 

from sklearn.metrics import accuracy_score, top_k_accuracy_score, precision_recall_curve as skl_prc
from sklearn.metrics import roc_curve as skl_roc, auc as skl_auc, confusion_matrix as skl_cm

import json, os, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from spanet.evaluation_scm import evaluate_on_test_dataset, load_model


# ------------------------------------------------------------------ CLASSIFIER
def classifier_metrics(class_truth: np.ndarray,
                       class_logits: np.ndarray,
                       class_pred: np.ndarray,
                       k: int) -> Dict[str, float]:
    """
    Return Top-1 plus a bunch of Top-m accuracies for m = 2 ... (2k-1).

    Parameters
    ----------
    class_truth  : (E, K) one-hot truth
    class_logits : (E, K) raw logits
    class_pred   : (E,)   argmax prediction per event
    k            : int    reference window size; report up to 2k-1
    """
    true_labels = class_truth.argmax(axis=1)
    num_classes = class_logits.shape[1]

    # Always include Top-1
    metrics = {
        "Top-1": accuracy_score(true_labels, class_pred)
    }

    # Add Top-m
    for m in range(2, min(2 * k, num_classes) + 1):
        metrics[f"Top-{m}"] = top_k_accuracy_score(
            true_labels,
            class_logits,
            k=m,
            labels=np.arange(num_classes)
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
        "Confusion": skl_cm(T, mask_pred.ravel()).tolist(),
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

# ------------------------------------------------------------- PHYSICS
def reco_mass(features: np.ndarray,
              class_pred: np.ndarray,
              branch: int) -> np.ndarray:
    """
    Reconstruct invariant mass of a given branch for each event.
    features : (E, K, B, J, F) with [mass, pt, eta, phi, ...]
    """
    jets = features[np.arange(features.shape[0]), class_pred, branch]   # (E,J,F)
    m, pt, eta, phi = jets[..., 0], jets[..., 1], jets[..., 2], jets[..., 3]
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E  = np.sqrt(m**2 + px**2 + py**2 + pz**2)

    E_sum  = E.sum(axis=1)
    px_sum = px.sum(axis=1)
    py_sum = py.sum(axis=1)
    pz_sum = pz.sum(axis=1)
    return np.sqrt(np.clip(E_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2, 0, None))

def total_pT(features: np.ndarray,
             class_pred: np.ndarray) -> np.ndarray:
    """Sum transverse momentum of every jet in the chosen hypothesis."""
    jets = features[np.arange(features.shape[0]), class_pred]  # (E,B,J,F)
    pt, phi = jets[..., 1], jets[..., 3]
    px, py  = pt * np.cos(phi), pt * np.sin(phi)
    return np.sqrt(px.sum(axis=(1,2))**2 + py.sum(axis=(1,2))**2)

def moments(arr):
    return (float(arr.mean()) if arr.size else np.nan,
            float(arr.std())  if arr.size else np.nan)

def physics_metrics(features   : np.ndarray,
                    class_pred : np.ndarray,
                    full_mask  : np.ndarray,
                    partial_mask: np.ndarray) -> Dict[str, float | np.ndarray]:
    """Top-mass & total‑pT moments for all / full / partial events."""
    m_top  = reco_mass(features, class_pred, branch=0)
    pT_tot = total_pT(features, class_pred)

    mt_mean,  mt_std  = moments(m_top)
    pt_mean,  pt_std  = moments(pT_tot)
    mt_full_mean, mt_full_std = moments(m_top[full_mask])
    mt_part_mean, mt_part_std = moments(m_top[partial_mask])
    pt_full_mean, pt_full_std = moments(pT_tot[full_mask])
    pt_part_mean, pt_part_std = moments(pT_tot[partial_mask])

    return {
        "Top_mass_mean"      : mt_mean,
        "Top_mass_std"       : mt_std,
        "pT_tot_mean"        : pt_mean,
        "pT_tot_std"         : pt_std,
        "Top_mass_mean_full" : mt_full_mean,
        "Top_mass_std_full"  : mt_full_std,
        "Top_mass_mean_partial": mt_part_mean,
        "Top_mass_std_partial" : mt_part_std,
        "pT_tot_mean_full"   : pt_full_mean,
        "pT_tot_std_full"    : pt_full_std,
        "pT_tot_mean_partial": pt_part_mean,
        "pT_tot_std_partial" : pt_part_std,
        # raw arrays returned for histograms
        "_m_top"       : m_top,
        "_pT_tot"      : pT_tot,
        "_m_top_full"  : m_top[full_mask],
        "_m_top_part"  : m_top[partial_mask],
        "_pT_full"     : pT_tot[full_mask],
        "_pT_part"     : pT_tot[partial_mask],
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

    m_phys  = physics_metrics(feats, CPd,
                              m_joint["_full_mask"],
                              m_joint["_partial_mask"])
    metrics.update({k:v for k,v in m_phys.items() if not k.startswith("_")})

    # curves for PDF
    recall, precision = m_mask["_pr_curve"]
    fpr, tpr          = m_mask["_roc_curve"]

    # arrays for physics histos
    m_top       = m_phys["_m_top"]
    pT_tot      = m_phys["_pT_tot"]
    m_top_full  = m_phys["_m_top_full"]
    m_top_partial = m_phys["_m_top_part"]
    pT_tot_full   = m_phys["_pT_full"]
    pT_tot_partial= m_phys["_pT_part"]


    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

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

        # Top mass
        plt.figure()
        plt.hist(m_top, bins=60)
        plt.xlabel(r"$m_\mathrm{reco}^{\mathrm{top}}\;[\mathrm{GeV}]$")
        plt.ylabel("Events")
        pdf.savefig(); plt.close()

        # Total pT
        plt.figure()
        plt.hist(pT_tot, bins=60)
        plt.xlabel(r"$p_T^{\mathrm{tot}}\;[\mathrm{GeV}]$")
        plt.ylabel("Events")
        pdf.savefig(); plt.close()

        # Plot mass: full
        plt.figure()
        plt.hist(m_top_full, bins=60, alpha=0.7, label="Full reco")
        plt.xlabel(r"$m_\mathrm{reco}^{\mathrm{top}}\;[\mathrm{GeV}]$")
        plt.ylabel("Events")
        plt.title("Reconstructed Top Mass")
        plt.legend()
        pdf.savefig(); plt.close()

        # Plot mass: partial
        plt.figure()
        plt.hist(m_top_partial, bins=60, alpha=0.7, label="Partial reco")
        plt.xlabel(r"$m_\mathrm{reco}^{\mathrm{top}}\;[\mathrm{GeV}]$")
        plt.ylabel("Events")
        plt.title("Reconstructed Top Mass")
        plt.legend()
        pdf.savefig(); plt.close()

        # Plot total pT: full
        plt.figure()
        plt.hist(pT_tot_full, bins=60, alpha=0.7, label="Full reco")
        plt.xlabel(r"$p_T^{\mathrm{tot}}\;[\mathrm{GeV}]$")
        plt.ylabel("Events")
        plt.title("Total $p_T$")
        plt.legend()
        pdf.savefig(); plt.close()

        # Plot total pT: partial
        plt.figure()
        plt.hist(pT_tot_partial, bins=60, alpha=0.7, label="Partial reco")
        plt.xlabel(r"$p_T^{\mathrm{tot}}\;[\mathrm{GeV}]$")
        plt.ylabel("Events")
        plt.title("Total $p_T$")
        plt.legend()
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