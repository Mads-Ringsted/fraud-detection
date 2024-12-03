from sklearn.cluster import SpectralClustering
from scipy.stats import mode
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics.pairwise import rbf_kernel
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from utils.scoring import clustering_classification_report, score_clustering



def format_clustering_metrics(train_scores, test_scores, **kwargs):
    results = kwargs
    results.update({
        'Train_DB': train_scores[0],
        'Train_Sil': train_scores[1],
        'Train_Pur': train_scores[2],
        'Test_DB': test_scores[0],
        'Test_Sil': test_scores[1],
        'Test_Pur': test_scores[2],
    })
    return results

def format_classification_metrics(train_metrics, test_metrics, **kwargs):
    results = kwargs
    results.update({
        'Train_Acc': train_metrics['accuracy'],
        'Train_F1': train_metrics['f1'],
        'Train_Recall': train_metrics['recall'],
        'Train_Precision': train_metrics['precision'],
        'Test_Acc': test_metrics['accuracy'],
        'Test_F1': test_metrics['f1'],
        'Test_Recall': test_metrics['recall'],
        'Test_Precision': test_metrics['precision'],
    })
    return results



def derive_cluster_mapping(y_train, train_cluster_labels):
    """
    Derives a mapping of cluster labels to actual class labels using the training set.
    """
    mapping = {}
    for cluster in np.unique(train_cluster_labels):
        mask = (train_cluster_labels == cluster)
        if np.any(mask):  # Ensure the mask selects at least one element
            mode_result = mode(y_train[mask], nan_policy='omit')  # Handle potential NaNs
            cluster_mode = mode_result.mode
            if isinstance(cluster_mode, np.ndarray) and cluster_mode.size > 0:
                mapping[cluster] = cluster_mode[0]  # Access the mode value if it's an array
            else:
                mapping[cluster] = cluster_mode  # Directly assign the scalar value
        else:
            raise ValueError(f"Cluster {cluster} is empty. Check clustering assignments.")
    return mapping

def map_clusters(cluster_labels, mapping):
    """
    Maps cluster labels to class labels based on the derived mapping.
    """
    return np.array([mapping[label] for label in cluster_labels])

def evaluate_spectral(X_train, X_test, y_train, y_test, n_clusters):
    # Step 1: Fit spectral clustering on training data
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='rbf', gamma=1.0, assign_labels='kmeans', random_state=42)
    train_cluster_labels = spectral.fit_predict(X_train)

    # Step 2: Derive cluster-to-class mapping from training data
    cluster_to_class_mapping = derive_cluster_mapping(y_train, train_cluster_labels)

    # Step 3: Assign test data to nearest training clusters
    # Approximate cluster centroids based on X_train clusters
    train_centroids = np.array([X_train[train_cluster_labels == i].mean(axis=0) for i in range(n_clusters)])
    test_cluster_labels = pairwise_distances_argmin(X_test, train_centroids, metric='euclidean')

    # Step 4: Map cluster labels to class labels for train and test data
    train_aligned_labels = map_clusters(train_cluster_labels, cluster_to_class_mapping)
    test_aligned_labels = map_clusters(test_cluster_labels, cluster_to_class_mapping)

    # Step 5: Compute clustering scores
    train_clustering_scores = score_clustering(X_train, y_train, train_aligned_labels)
    test_clustering_scores = score_clustering(X_test, y_test, test_aligned_labels)

    # Step 6: Compute classification scores
    train_classification_report = clustering_classification_report(train_aligned_labels, y_train)
    test_classification_report = clustering_classification_report(test_aligned_labels, y_test)

    # Step 7: Format metrics
    clustering_metrics = format_clustering_metrics(train_clustering_scores, test_clustering_scores, n_clusters=n_clusters)
    classification_metrics = format_classification_metrics(train_classification_report, test_classification_report, n_clusters=n_clusters)

    return clustering_metrics, classification_metrics

def score_spectral(X_train, X_test, y_train, y_test, cluster_counts=None):
    if cluster_counts is None:
        cluster_counts = [2, 3, 5, 10]
    clustering_metrics_list = []
    classification_metrics_list = []
    for n_clusters in cluster_counts:
        clustering_scores, classification_scores = evaluate_spectral(X_train, X_test, y_train, y_test, n_clusters)
        clustering_metrics_list.append(clustering_scores)
        classification_metrics_list.append(classification_scores)

    return clustering_metrics_list, classification_metrics_list


def extract_and_visualize_graph(X_train, y_train=None, affinity='rbf', gamma=1.0, n_neighbors=10):
    """
        Extracts and visualizes the graph used in spectral clustering, using X_train.

        Parameters:
            X_train (np.ndarray): Training data features.
            y_train (np.ndarray): Optional training labels for coloring nodes (default=None).
            affinity (str): Type of graph to extract ('rbf' or 'nearest_neighbors').
            gamma (float): Parameter for the RBF kernel (if affinity='rbf').
            n_neighbors (int): Number of neighbors (if affinity='nearest_neighbors').

        Returns:
            graph (nx.Graph): NetworkX graph representation of the data.
            plt (matplotlib.pyplot): Matplotlib plot object for further customization.
        """
    if affinity == 'rbf':
        # Compute RBF (Gaussian) kernel
        affinity_matrix = rbf_kernel(X_train, gamma=gamma)
    elif affinity == 'nearest_neighbors':
        # Compute k-nearest neighbors graph
        affinity_matrix = kneighbors_graph(X_train, n_neighbors=n_neighbors, include_self=False).toarray()
    else:
        raise ValueError("Affinity must be 'rbf' or 'nearest_neighbors'")

    # Create a NetworkX graph
    graph = nx.from_numpy_array(affinity_matrix)

    # Visualize the graph
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(graph)  # Layout for visualization
    nx.draw_networkx_nodes(graph, pos, node_size=50, node_color=y_train if y_train is not None else 'gray', cmap=plt.cm.coolwarm)
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    plt.title(f"Graph Visualization ({affinity} affinity)")

    return graph, plt
