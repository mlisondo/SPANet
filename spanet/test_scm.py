from typing import Optional

from argparse import ArgumentParser

import numpy as np
import torch                                 

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





def main(log_directory, test_file, event_file,
         batch_size, gpu, fp16, top_k, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # ---------------- load model (our dual‑head) ----------------
    model = SCM_Eval_Test.load_from_checkpoint(
        os.path.join(log_directory, "checkpoints", "last.ckpt"),
        strict=False,
        options=None,                 # will be overridden by Lightning checkpoint
        class_hidden_dims=[30, 64],
        mask_hidden_dims=[30, 64],
        torch_script=False
    ).eval()

    if top_k is not None:
        model.options.k = top_k

    device = "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
    model.to(device)

    loader = DataLoader(model.testing_dataset,
                        batch_size=batch_size or model.options.batch_size,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True)

    # --------------- accumulate batch‑wise outputs ---------------
    all_CL, all_CT, all_CPd = [], [], []
    all_ML, all_MT, all_MPd = [], [], []
    all_features = []

    for batch in loader:
        batch = [x.to(device) if torch.is_tensor(x) else x for x in batch]
        out = model.evaluate_batch(batch)

        all_CL.append(out["class_logits"].cpu().numpy())
        all_CT.append(out["class_truth"].cpu().numpy())
        all_CPd.append(out["class_preds"].cpu().numpy())

        all_ML.append(out["mask_logits"].cpu().numpy())
        all_MT.append(out["mask_truth"].cpu().numpy())
        all_MPd.append(out["mask_preds"].cpu().numpy())

        all_features.append(out["features_arr"].cpu().numpy())

    CL = np.concatenate(all_CL)
    CT = np.concatenate(all_CT)
    CPd = np.concatenate(all_CPd)

    ML = np.concatenate(all_ML)
    MT = np.concatenate(all_MT)
    MPd = np.concatenate(all_MPd)

    feats = np.concatenate(all_features)

    # ------------------ numeric metrics ------------------
    metrics = {}
    metrics["Top-1"] = float(top1_acc(CT, CPd))
    metrics["Top-{}".format(model.options.k)] = float(topk_acc(CT, CL, model.options.k))

    # masker: flatten (E,K,B) → (N,)
    MP   = 1/(1+np.exp(-ML))          # sigmoid after concatenation
    precision, recall = precision_recall_curve(MP.ravel(), MT.ravel())
    tpr, fpr = roc_curve(MP.ravel(), MT.ravel())
    metrics["AUC_PR"]  = float(auc(recall, precision))
    metrics["AUC_ROC"] = float(auc(fpr, tpr))
    metrics["Confusion"] = confusion_matrix(MPd, MT).tolist()

    # joint
    metrics["Event_eff"]   = float(EC(CT, CPd, MPd, MT))
    metrics["Partial_rec"] = float(PR(CT, CPd, MPd, MT))

    # physics (example: top‑mass from branch 0, total‑pT)
    m_top   = reco_mass(feats, CPd, branch=0)
    pT_tot  = total_pT(feats, CPd)
    metrics["Top_mass_mean"] = float(m_top.mean())
    metrics["pT_tot_mean"]   = float(pT_tot.mean())

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # ------------------ figures ------------------
    with PdfPages(os.path.join(output_dir, "plots.pdf")) as pdf:
        # PR curve
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision‑Recall")
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
