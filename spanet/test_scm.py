from typing import Optional

from argparse import ArgumentParser

import numpy as np
import torch                                 

from sklearn.metrics import accuracy_score, top_k_accuracy_score, precision_recall_curve as skl_prc
from sklearn.metrics import roc_curve as skl_roc, auc as skl_auc, confusion_matrix as skl_cm

import json, os, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from spanet.evaluation_scm import evaluate_on_test_dataset, load_model


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
        k=k,
        labels=np.arange(class_logits.shape[1]) # Ensures all possible class labels are considered
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

    # ------------------ numeric metrics ------------------
    metrics = {}

    # classifier
    metrics["Top-1"] = float(top1_acc(CT, CPd))
    metrics["Top-{}".format(model.options.k)] = float(topk_acc(CT, CL, model.options.k))

    # masker: flatten (E,K,B) -> (N,)
    precision, recall = precision_recall_curve(MP.ravel(), MT.ravel())
    fpr, tpr = roc_curve(MP.ravel(), MT.ravel())
    metrics["AUC_PR"]  = float(auc(recall, precision))
    metrics["AUC_ROC"] = float(auc(fpr, tpr))
    metrics["Confusion"] = confusion_matrix(MPd.ravel(), MT.ravel()).tolist()  # Flatten for confusion

    # joint
    metrics["Event_eff"]   = float(EC(CT, CPd, MPd, MT))
    metrics["Partial_rec"] = float(PR(CT, CPd, MPd, MT))

    # physics (example: to-mass from branch 0, total-pT)
    m_top   = reco_mass(feats, CPd, branch=0)
    pT_tot  = total_pT(feats, CPd)
    metrics["Top_mass_mean"] = float(m_top.mean())
    metrics["pT_tot_mean"]   = float(pT_tot.mean())

    # physics (fully vs partial)
    E, K, B = MPd.shape
    event_idx = np.arange(E)
    is_correct_class = (CT[event_idx, CPd] == 1)
    matches = (MPd[event_idx, CPd, :] == MT[event_idx, CPd, :])  # (E, B)
    n_correct_branches = np.sum(matches, axis=1)
    is_full_mask    = (n_correct_branches == B)
    is_partial_mask = (n_correct_branches > 0) & (n_correct_branches < B)

    fully_reco_mask   = is_correct_class & is_full_mask
    partial_reco_mask = is_correct_class & is_partial_mask

    m_top_full      = m_top[fully_reco_mask]
    m_top_partial   = m_top[partial_reco_mask]
    pT_tot_full     = pT_tot[fully_reco_mask]
    pT_tot_partial  = pT_tot[partial_reco_mask]

    metrics["n_full_reco"] = int(fully_reco_mask.sum())
    metrics["n_partial_reco"] = int(partial_reco_mask.sum())

    metrics["Top_mass_mean_full"] = float(m_top_full.mean()) if len(m_top_full) > 0 else float('nan')
    metrics["Top_mass_std_full"]  = float(m_top_full.std()) if len(m_top_full) > 0 else float('nan')
    metrics["Top_mass_mean_partial"] = float(m_top_partial.mean()) if len(m_top_partial) > 0 else float('nan')
    metrics["Top_mass_std_partial"]  = float(m_top_partial.std()) if len(m_top_partial) > 0 else float('nan')

    metrics["pT_tot_mean_full"] = float(pT_tot_full.mean()) if len(pT_tot_full) > 0 else float('nan')
    metrics["pT_tot_std_full"]  = float(pT_tot_full.std())  if len(pT_tot_full) > 0 else float('nan')
    metrics["pT_tot_mean_partial"] = float(pT_tot_partial.mean()) if len(pT_tot_partial) > 0 else float('nan')
    metrics["pT_tot_std_partial"]  = float(pT_tot_partial.std())  if len(pT_tot_partial) > 0 else float('nan')

    # 1) Are MP really probabilities?
    print("MP min/max/mean:", MP.min(), MP.max(), MP.mean())

    # 2) Compare score distributions by label
    pos = MP[MT == 1]; neg = MP[MT == 0]
    print("pos mean:", pos.mean(), "neg mean:", neg.mean(), "pos>neg?", pos.mean() > neg.mean())

    # 3) Try flipped scores
    from sklearn.metrics import roc_auc_score, average_precision_score
    print("ROC AUC (as-is):", roc_auc_score(MT.ravel(), MP.ravel()))
    print("ROC AUC (flipped):", roc_auc_score(MT.ravel(), 1.0 - MP.ravel()))
    print("AP (as-is):", average_precision_score(MT.ravel(), MP.ravel()))
    print("AP (flipped):", average_precision_score(MT.ravel(), 1.0 - MP.ravel()))

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

        # Plot mass: full vs partial
        plt.figure()
        plt.hist(m_top_full, bins=60, alpha=0.7, label="Full reco")
        plt.hist(m_top_partial, bins=60, alpha=0.7, label="Partial reco")
        plt.xlabel(r"$m_\mathrm{reco}^{\mathrm{top}}\;[\mathrm{GeV}]$")
        plt.ylabel("Events")
        plt.title("Reconstructed Top Mass")
        plt.legend()
        pdf.savefig(); plt.close()

        # Plot total pT: full vs partial
        plt.figure()
        plt.hist(pT_tot_full, bins=60, alpha=0.7, label="Full reco")
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