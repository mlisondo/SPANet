from typing import Optional

from argparse import ArgumentParser

import numpy as np
import torch                                 

from spanet.evaluation import evaluate_on_test_dataset, load_model

from sklearn.metrics import accuracy_score, top_k_accuracy_score, precision_recall_curve as skl_prc
from sklearn.metrics import roc_curve as skl_roc, auc as skl_auc, confusion_matrix as skl_cm

import json, os, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from spanet.network.jet_reconstruction.jet_scm_eval_test import SCM_Eval_Test
from torch.utils.data import DataLoader


# ---------------------- Classifier ----------------------



def top1_acc(class_truth, class_pred): # Top-1 accuracy: fraction of events where predicted hypothesis is correct
    """
    class_truth: (E, K)
    class_pred: (E,)
    """
    true_labels = np.argmax(class_truth, axis=1) # Convert one-hot to integer class labels
    return accuracy_score(true_labels, class_pred) # float: top-1 accuracy over batch

def topk_acc(class_truth, class_logits, k): # Top-K accuracy: correct hypo in top-k hypotheses by score.
    """
    class_truth: (E, K)
    class_logits: (E, K)
    """
    true_labels = np.argmax(class_truth, axis=1)
    return top_k_accuracy_score(
        true_labels,    # the correct class index per event
        class_logits,   # score array
        k,
        np.arange(class_logits.shape[1]) # Ensures all possible class labels are considered
    )



# ---------------------- Masker ----------------------



def precision_recall_curve(P, T): # Compute precision-recall curve for mask probabilities using sklearn.
    """
    P: (N,) mask probabilities (flattened)
    T: (N,) mask truth (flattened, 0/1)
    """
    precision, recall, _ = skl_prc(T, P)
    return precision, recall # precision, recall (all np.arrays)

def roc_curve(P, T): # Compute ROC curve for mask probabilities using sklearn.
    """
    P: (N,) mask probabilities (flattened)
    T: (N,) mask truth (flattened, 0/1)
    """
    fpr, tpr, _ = skl_roc(T, P)
    return tpr, fpr # tpr, fpr

def auc(x, y):
    return skl_auc(x, y)

def confusion_matrix(mask_pred, mask_truth):
    """
    mask_pred: (...,) binary 0/1
    mask_truth: (...,) binary 0/1
    """
    return skl_cm(mask_truth.ravel(), mask_pred.ravel()) # Order: [[tn, fp], [fn, tp]]



# ---------------------- Joint ----------------------



def EC(class_truth, class_pred, mask_pred, mask_truth): # Event-level reconstruction efficiency: correct class and all masks correct
    """
    class_truth: (E, K)
    class_pred: (E,)
    mask_pred: (E, K, B) binary
    mask_truth: (E, K, B) binary
    """
    E, _ = class_truth.shape
    correct_class = class_truth[np.arange(E), class_pred] == 1
    # For each event: are ALL branches of chosen hypo correct?
    all_branches = np.all(mask_pred[np.arange(E), class_pred, :] == mask_truth[np.arange(E), class_pred, :], axis=1)
    return np.mean(correct_class & all_branches) # efficiency

def PR(class_truth, class_pred, mask_pred, mask_truth): # Fraction of partial reconstructions: correct class and at least one branch correct
    """
    Args:
    class_truth: (E, K)
    class_pred: (E,)
    mask_pred: (E, K, B)
    mask_truth: (E, K, B)
    """
    E, _ = class_truth.shape
    B = mask_pred.shape[2]
    correct_class = class_truth[np.arange(E), class_pred] == 1
    matches = mask_pred[np.arange(E), class_pred, :] == mask_truth[np.arange(E), class_pred, :]
    n_correct_branches = np.sum(matches, axis=1)
    # At least one branch right, but not all
    partial = (n_correct_branches > 0) & (n_correct_branches < B)
    return np.mean(correct_class & partial) # partial reconstruction fraction



# ---------------------- Physics ----------------------



def reco_mass(features, class_pred, branch): # Reconstruct mass for each event's assigned hypothesis and branch
    """
    features: (E, K, B, J, F)
    class_pred: (E,)
    branch: e.g. 0 for top, 1 for tbar
    """
    jets = features[np.arange(features.shape[0]), class_pred, branch]  # (E, J, F)
    # Features: [mass, pt, eta, phi, btag]      NEED TO CHECK
    eta = jets[..., 2]
    mass = jets[..., 0]
    phi = jets[..., 3]
    pt = jets[..., 1]
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E_jet = np.sqrt(mass**2 + px**2 + py**2 + pz**2)
    # Sum 4-vectors over all jets for each event
    E_sum = np.sum(E_jet, axis=1)
    px_sum = np.sum(px, axis=1)
    py_sum = np.sum(py, axis=1)
    pz_sum = np.sum(pz, axis=1)
    mass_reco = np.sqrt(np.clip(E_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2, 0, None))
    return mass_reco  # For histogram/plot

def total_pT(features, class_pred): # Reconstructed total transverse momentum for each event (sum all branches/jets)
    """
    features: (E, K, B, J, F)
    class_pred: (E,)
    """
    # Take all jets in all branches for assigned hypo
    E, _, _, _, _ = features.shape
    jets = features[np.arange(E), class_pred]  # (E, B, J, F)
    phi = jets[..., 3]  # (E, B, J,)
    pt = jets[..., 1]
    px = pt * np.cos(phi)   # (E, B, J,)
    py = pt * np.sin(phi)
    px_tot = np.sum(px, axis=(1, 2))  # sum over branches and jets
    py_tot = np.sum(py, axis=(1, 2))
    pT_tot = np.sqrt(px_tot**2 + py_tot**2)
    return pT_tot  # total_pT: (E,) for histogram/plot

def mass_window_efficiency(masses, min_mass, max_mass): # Fraction of events with mass in window
    """
    masses: (E,)
    min_mass, max_mass: window radius
    """
    return np.mean((masses > min_mass) & (masses < max_mass))





# ---------------------- MAIN ----------------------
