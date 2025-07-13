#!/usr/bin/env python3
"""
cluster_sst_analysis.py - Cluster analysis of normalized SST data for TS & HU fixes

This script:
1. Loads SST data using load_windows function
2. Normalizes each row (z-score normalization)
3. Performs K-means clustering for k=2, 3, 4
4. Plots cluster means in three subplots along with overall mean

Panels
------
a : 2-cluster analysis
b : 3-cluster analysis  
c : 4-cluster analysis

Each panel shows cluster means and overall mean against days from storm passage.
Figure saved as *sst_cluster_analysis.png* and *.pdf* and displayed.

Usage
-----
$ python cluster_sst_analysis.py            # scans ./t_data
$ python cluster_sst_analysis.py /path/to/t_data
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sst_loader import load_windows

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

# ─────────────────────────────────────────────────────────────────────────────
# Normalization function
# ─────────────────────────────────────────────────────────────────────────────

def normalize_rows(data: np.ndarray) -> np.ndarray:
    """
    Normalize each row using z-score normalization (subtract mean, divide by std).
    
    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
        
    Returns
    -------
    np.ndarray
        Normalized data with same shape as input
    """
    normalized = np.zeros_like(data)
    for i in range(data.shape[0]):
        row = data[i]
        # Only normalize if we have valid data and non-zero std
        valid_mask = np.isfinite(row)
        if valid_mask.sum() > 1:
            row_mean = np.nanmean(row)
            row_std = np.nanstd(row)
            if row_std > 0:
                normalized[i] = (row - row_mean) / row_std
            else:
                normalized[i] = row - row_mean
        else:
            normalized[i] = row
    return normalized

# ─────────────────────────────────────────────────────────────────────────────
# Clustering function
# ─────────────────────────────────────────────────────────────────────────────

def perform_clustering(data: np.ndarray, n_clusters: int, random_state: int = 42):
    """
    Perform K-means clustering on the data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    n_clusters : int
        Number of clusters
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    labels : np.ndarray
        Cluster labels for each sample
    cluster_centers : np.ndarray
        Cluster centers
    """
    if not HAVE_SKLEARN:
        raise ImportError("scikit-learn is required for clustering")
    
    # Remove rows with too many NaN values
    valid_rows = np.isfinite(data).sum(axis=1) >= data.shape[1] * 0.5
    valid_data = data[valid_rows]
    
    if valid_data.shape[0] < n_clusters:
        raise ValueError(f"Not enough valid samples ({valid_data.shape[0]}) for {n_clusters} clusters")
    
    # Fill remaining NaN values with row means
    for i in range(valid_data.shape[0]):
        row = valid_data[i]
        if np.isnan(row).any():
            row_mean = np.nanmean(row)
            valid_data[i] = np.where(np.isnan(row), row_mean, row)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels_valid = kmeans.fit_predict(valid_data)
    
    # Map back to original data size
    labels = np.full(data.shape[0], -1)
    labels[valid_rows] = labels_valid
    
    return labels, kmeans.cluster_centers_

# ─────────────────────────────────────────────────────────────────────────────
# Plotting function
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_means(ax, data: np.ndarray, labels: np.ndarray, n_clusters: int, 
                      days: np.ndarray, panel: str):
    """
    Plot cluster means and overall mean.
    """
    # Overall mean
    overall_mean = np.nanmean(data, axis=0)
    ax.plot(days, overall_mean, 'k-', linewidth=2, label='Overall mean', alpha=0.8)
    
    # Cluster means
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown'][:n_clusters]
    
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        if cluster_mask.sum() > 0:
            cluster_data = data[cluster_mask]
            cluster_mean = np.nanmean(cluster_data, axis=0)
            n_samples = cluster_mask.sum()
            ax.plot(days, cluster_mean, color=colors[cluster_id], 
                   linewidth=1.5, label=f'Cluster {cluster_id + 1} (n={n_samples})')
    
    ax.set_xlabel('Days from storm passage')
    ax.set_ylabel('Normalized SST')
    ax.set_title(f'{n_clusters}-cluster analysis', fontsize=11)
    ax.axvline(0, color='k', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Panel label
    ax.text(0.02, 0.96, f'$\\mathbf{{{panel}}}$', transform=ax.transAxes,
            ha='left', va='top', fontsize=12)

# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not HAVE_SKLEARN:
        sys.exit("✗ scikit-learn is required. Install with: pip install scikit-learn")
    
    # Load data
    t_data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('t_data')
    if not t_data_dir.is_dir():
        sys.exit(f"✗ Directory '{t_data_dir}' not found")
    
    print("Loading SST data...")
    data = load_windows(t_data_dir)  # Shape: (n_samples, 31)
    print(f"✓ Loaded {data.shape[0]} SST windows")
    
    # Normalize data
    print("Normalizing data...")
    normalized_data = normalize_rows(data)
    
    # Days array (31 days: -15 to +15)
    days = np.arange(-15, 16)
    
    # Perform clustering for k=2, 3, 4
    cluster_results = {}
    for k in [2, 3, 4]:
        print(f"Performing {k}-means clustering...")
        try:
            labels, centers = perform_clustering(normalized_data, k)
            cluster_results[k] = labels
            print(f"✓ {k}-cluster analysis complete")
        except Exception as e:
            print(f"✗ Error in {k}-cluster analysis: {e}")
            cluster_results[k] = None
    
    # Create the plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    panel_labels = ['a', 'b', 'c']
    cluster_counts = [2, 3, 4]
    
    for i, (ax, panel, k) in enumerate(zip(axes, panel_labels, cluster_counts)):
        if cluster_results[k] is not None:
            plot_cluster_means(ax, normalized_data, cluster_results[k], k, days, panel)
        else:
            ax.text(0.5, 0.5, f'Clustering failed\nfor k={k}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{k}-cluster analysis (failed)')
    
    plt.tight_layout()
    
    # Save the figure
    for ext in ('png', 'pdf'):
        fig.savefig(Path(f'sst_cluster_analysis.{ext}'), dpi=300, bbox_inches='tight')
    
    print('✓ Figure saved as sst_cluster_analysis.png and .pdf')
    plt.show()

if __name__ == '__main__':
    main()