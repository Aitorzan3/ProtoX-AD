import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import torch
import torch.nn
from torch import nn
import random
import torch.nn.functional as F
from transformations import *

device = torch.device("mps")

# Function to find the optimal threshold for maximizing F1-score and compute FPR, FNR after the optimal threshold is found
def metrics(anomaly_scores, true_labels):
    true_labels = 1 - true_labels

    # Compute precision, recall, and thresholds
    precision, recall, _ = precision_recall_curve(true_labels, anomaly_scores)

    # Compute AUC-PR using the trapezoidal rule
    auc_pr = auc(recall, precision)
    
    # Compute AUC-ROC
    auc_roc = roc_auc_score(true_labels, anomaly_scores)
    
    return auc_roc, auc_pr

# Method to set the seed and ensure reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.Generator().manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def compute_latents(loader, model):
    mus = []
    net = model._net if hasattr(model, "_net") else model
    for batch in loader:
        batch = batch.to(device).float()
        mu, _  = net.encoder(batch)                   # (K*B,T,D)

        mus.append(mu.cpu().numpy())

    mus = np.concatenate(mus, axis=0)
    return mus

from dtaidistance import dtw

@torch.no_grad()
def nearest_prototypes_global(mu, model):
    net = model._net if hasattr(model, "_net") else model
    proto = net.prototype_vectors.cpu()# (P,D)
    sims = torch.cdist(torch.tensor(mu), proto)                      # (M,P)
    nearest_idx = sims.argmin(axis=1)
    nearest_sim = sims[np.arange(sims.shape[0]), nearest_idx]
    return nearest_idx, nearest_sim

from scipy.spatial.distance import euclidean

def compute_distances_to_nearest_prototypes(dataloader, nearest_match, proto_series):
    if isinstance(proto_series, np.ndarray):
        proto_series_t = torch.tensor(proto_series, dtype=torch.float32)
    else:
        proto_series_t = proto_series.clone().detach().float()

    l1_distances = []
    l2_distances = []

    idx_global = 0

    for batch in dataloader:
        batch = batch.squeeze(0) if batch.dim() == 4 else batch  # por si viene (1,B,T,F)
        batch = batch.float()

        B = batch.shape[0]

        for b in range(B):
            proto_idx = nearest_match[idx_global]
            proto = proto_series_t[proto_idx]      # (T,F)
            sample = batch[b]                      # (T,F)

            # ---- L1 ----
            dist_l1 = F.l1_loss(sample, proto, reduction='mean').item()
            l1_distances.append(dist_l1)

            # ---- L2 ----
            dist_l2 = F.mse_loss(sample, proto, reduction='mean').sqrt().item()
            l2_distances.append(dist_l2)

            idx_global += 1

    # Convert to numpy
    l1_np = np.array(l1_distances)
    l2_np = np.array(l2_distances)

    results = {
        "l1_distances": l1_distances,
        "l2_distances": l2_distances,
        "l1_mean": l1_np.mean(),
        "l2_mean": l2_np.mean(),
        "l1_std": l1_np.std(),
        "l2_std": l2_np.std(),
    }

    print("---- Distances ----")
    print(f"L1  mean: {results['l1_mean']:.6f} | std: {results['l1_std']:.6f}")
    print(f"L2  mean: {results['l2_mean']:.6f} | std: {results['l2_std']:.6f}")
    return results['l1_mean'], results['l2_mean'], results['l1_std'], results['l2_std']

def plot_samples_with_nearest_prototypes(
    samples,
    nearest_match,
    proto_series,
    sample_ids=None,
    labels=None,
    transform_names=None,
    num_prototypes_per_class=None,
    feature=0,
    n_examples=4,
    ylim=None,
    figsize=(12, 8),
):
    """
    Plot selected samples together with their nearest input-space prototypes.

    Parameters
    ----------
    samples : array-like or torch.Tensor
        Test samples with shape (N, T, F) or (N, T).
    nearest_match : array-like
        Index of the nearest prototype for each sample.
    proto_series : array-like or torch.Tensor
        Reconstructed input-space prototypes with shape (P, T, F).
    sample_ids : list or array-like, optional
        Indices of the samples to plot. If None, the first n_examples are used.
    labels : array-like, optional
        Ground-truth labels for the samples.
    transform_names : list, optional
        Names of the transformation classes.
    num_prototypes_per_class : int, optional
        Number of prototypes per transformation class. Required to infer the matched class.
    feature : int
        Feature/channel to plot.
    n_examples : int
        Number of examples to plot if sample_ids is None.
    ylim : tuple, optional
        y-axis limits, e.g. (-2, 2).
    figsize : tuple
        Figure size.
    """

    if isinstance(samples, torch.Tensor):
        samples_np = samples.detach().cpu().numpy()
    else:
        samples_np = np.asarray(samples)

    if isinstance(proto_series, torch.Tensor):
        proto_np = proto_series.detach().cpu().numpy()
    else:
        proto_np = np.asarray(proto_series)

    nearest_match = np.asarray(nearest_match)

    if samples_np.ndim == 2:
        samples_np = samples_np[:, :, None]

    if proto_np.ndim == 2:
        proto_np = proto_np[:, :, None]

    if sample_ids is None:
        sample_ids = np.arange(min(n_examples, len(samples_np)))
    else:
        sample_ids = np.asarray(sample_ids)

    n_rows = len(sample_ids)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True, squeeze=False)

    for row, sample_idx in enumerate(sample_ids):
        ax = axes[row, 0]

        proto_idx = int(nearest_match[sample_idx])
        sample = samples_np[sample_idx, :, feature]
        proto = proto_np[proto_idx, :, feature]

        t = np.arange(len(sample))

        ax.plot(t, sample, linewidth=1.8, label="Test sample")
        ax.plot(t, proto, linewidth=1.8, linestyle="--", label=f"Nearest prototype #{proto_idx}")

        title = f"Sample {sample_idx}"

        if labels is not None:
            title += f" | true label: {labels[sample_idx]}"

        if num_prototypes_per_class is not None:
            matched_class = proto_idx // num_prototypes_per_class
            if transform_names is not None and matched_class < len(transform_names):
                title += f" | matched pattern: {transform_names[matched_class]}"
            else:
                title += f" | matched class: {matched_class}"

        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, frameon=False)

        if ylim is not None:
            ax.set_ylim(*ylim)

    fig.supxlabel("Time index")
    fig.supylabel("Amplitude")
    fig.suptitle("Test samples and their nearest prototypes", fontsize=14, y=1.02)
    fig.tight_layout()

    return fig