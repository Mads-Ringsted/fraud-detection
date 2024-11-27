
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np


def plot_2d_clusters(X, y, cluster_labels, cluster_centroids=None):
    """
    Plot 2D clusters with clear and dynamic legends for fraud/no-fraud and cluster assignments.
    
    Parameters:
        X (np.ndarray): 2D array of shape (n_samples, 2) with features.
        y (np.ndarray): 1D binary array indicating fraud (0 or 1).
        cluster_labels (np.ndarray): 1D array indicating cluster assignment.
        cluster_centroids (np.ndarray): 2D array of cluster centroids (optional).
    """
    plt.figure(figsize=(10, 6))
    fraud_palette = {0: "blue", 1: "red"}  # Fraud/No-Fraud color palette

    # Create cluster markers dynamically
    unique_clusters = np.unique(cluster_labels)
    cluster_markers = ['o', 's', 'X', 'P', '*', 'D', '^', '+', '<', '>']  # Marker styles for clusters
    marker_map = {cluster: cluster_markers[i % len(cluster_markers)] for i, cluster in enumerate(unique_clusters)}

    # Plot points by clusters
    for cluster in unique_clusters:
        mask = cluster_labels == cluster
        fraud_mask = y[mask]
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            c=[fraud_palette[f] for f in fraud_mask],
            marker=marker_map[cluster],
            label=f"Cluster {cluster}" if cluster != -1 else "Outliers (-1)",
            s=50,
            edgecolor="k",  # Add border to markers for better clarity
        )

    # Add fraud/no-fraud legend manually
    fraud_legend = [
        mpatches.Patch(color="blue", label="No Fraud"),
        mpatches.Patch(color="red", label="Fraud"),
    ]

    # Combine fraud legend with cluster labels dynamically
    cluster_handles = [
        mlines.Line2D([], [], color="black", marker=marker_map[cluster], linestyle="None", markersize=10,
                      label=f"Cluster {cluster}" if cluster != -1 else "Outliers (-1)")
        for cluster in unique_clusters
    ]
    handles = fraud_legend + cluster_handles

    if cluster_centroids is not None:
        plt.scatter(
            cluster_centroids[:, 0],
            cluster_centroids[:, 1],
            c="yellow",  # Centroid color
            marker="D",  # Diamond shape for centroids
            s=200,
            edgecolor="black",
            label="Centroids",
        )
        centroid_handle = mlines.Line2D([], [], color="yellow", marker="D", linestyle="None", markersize=12, label="Centroids")
        handles += [centroid_handle]

    plt.legend(handles=handles, title="Legend", loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the plot area to accommodate the legend

    # Plot enhancements
    plt.title("2D Cluster Visualization with Clear Legend", fontsize=16)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()