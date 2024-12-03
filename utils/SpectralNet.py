import torch
from spectralnet import SpectralNet
from scipy.stats import mode
import numpy as np
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


def evaluate_spectral_net(X_train, X_test, y_train, y_test, n_clusters=None):
    # Define the SpectralNet model
    spectralnet = SpectralNet(n_clusters=n_clusters, spectral_hiddens=[128, 64, 16, n_clusters])

    # Convert training data to torch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()

    # Train the SpectralNet model on training data
    spectralnet.fit(X_train_tensor)

    # Predict cluster labels for training data
    with torch.no_grad():
        train_cluster_labels = spectralnet.predict(X_train_tensor)

    # Map clusters to class labels using training data
    def derive_cluster_mapping(y_true, cluster_labels):
        mapping = {}
        for cluster in np.unique(cluster_labels):
            labels_in_cluster = y_true[cluster_labels == cluster]
            if len(labels_in_cluster) == 0:
                continue
            mode_result = mode(labels_in_cluster)
            most_common = mode_result.mode.item()
            mapping[cluster] = most_common
        return mapping

    cluster_to_class_mapping = derive_cluster_mapping(y_train, train_cluster_labels)

    # Align training labels
    train_aligned_labels = np.array([cluster_to_class_mapping[label] for label in train_cluster_labels])

    # Predict cluster labels for test data
    with torch.no_grad():
        test_cluster_labels = spectralnet.predict(X_test_tensor)

    # Align test labels
    test_aligned_labels = np.array([cluster_to_class_mapping.get(label, -1) for label in test_cluster_labels])

    # Evaluate performance
    train_classification_report = clustering_classification_report(train_aligned_labels, y_train)
    test_classification_report = clustering_classification_report(test_aligned_labels, y_test)

    train_clustering_scores = score_clustering(X_train, y_train, train_aligned_labels)
    test_clustering_scores = score_clustering(X_test, y_test, test_aligned_labels)

    clustering_metrics = format_clustering_metrics(train_clustering_scores, test_clustering_scores, n_clusters=n_clusters)
    classification_metrics = format_classification_metrics(train_classification_report, test_classification_report, n_clusters=n_clusters)

    return clustering_metrics, classification_metrics

def score_spectral_net(X_train, X_test, y_train, y_test, cluster_counts=None):
    if cluster_counts is None:
        cluster_counts = [2, 3, 5, 10]
    clustering_metrics_list = []
    classification_metrics_list = []
    for n_clusters in cluster_counts:
        clustering_scores, classification_scores = evaluate_spectral_net(X_train, X_test, y_train, y_test, n_clusters)
        clustering_metrics_list.append(clustering_scores)
        classification_metrics_list.append(classification_scores)

    return clustering_metrics_list, classification_metrics_list


