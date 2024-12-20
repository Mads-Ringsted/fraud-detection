import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import silhouette_score, davies_bouldin_score
from utils.scoring import clustering_classification_report, score_clustering

####################################################################
# Genral Utils for Clustering                                      #
####################################################################

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


####################################################################
# KMeans Utils                                                     #
####################################################################

def fit_kmeans(X_train, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train)
    return kmeans

def extract_kmeans_cluster_labels(kmeans, X):
    return kmeans.predict(X)

def evaluate_kmeans(X_train, X_test, y_train, y_test, n_clusters):
    kmeans = fit_kmeans(X_train, n_clusters)
    train_cluster_labels = extract_kmeans_cluster_labels(kmeans, X_train)
    test_cluster_labels = extract_kmeans_cluster_labels(kmeans, X_test)

    train_clustering_scores = score_clustering(X_train, y_train, train_cluster_labels)
    test_clustering_scores = score_clustering(X_test, y_test, test_cluster_labels)

    train_classification_report = clustering_classification_report(train_cluster_labels, y_train)
    test_classification_report = clustering_classification_report(test_cluster_labels, y_test)
    clustering_metrics = format_clustering_metrics(train_clustering_scores, test_clustering_scores, n_clusters=n_clusters)
    classification_metrics = format_classification_metrics(train_classification_report, test_classification_report, n_clusters=n_clusters)
    return clustering_metrics, classification_metrics

def score_kmeans(X_train, X_test, y_train, y_test, cluster_counts=None):
    if cluster_counts is None:
        cluster_counts = [2, 3, 5, 10]
    clustering_metrics_list = []
    classification_metrics_list = []
    for n_clusters in cluster_counts:
        clustering_scores, classification_scores = evaluate_kmeans(X_train, X_test, y_train, y_test, n_clusters)
        clustering_metrics_list.append(clustering_scores)
        classification_metrics_list.append(classification_scores)
    return clustering_metrics_list, classification_metrics_list


####################################################################
# DBSCAN Utils                                                     #
####################################################################

def fit_dbscan(X_train, eps):
    dbscan = DBSCAN(eps=eps, min_samples=5)
    dbscan.fit(X_train)
    return dbscan

def test_dbscan(dbscan, X_train, X_test):
    core_samples_mask = dbscan.core_sample_indices_
    core_points = X_train[core_samples_mask]

    nn = NearestNeighbors(n_neighbors=1).fit(core_points)
    distances, indices = nn.kneighbors(X_test)

    test_clusters = np.array([dbscan.labels_[core_samples_mask[i]] if distances[j] < dbscan.eps else -1 
                            for j, i in enumerate(indices.flatten())])
    return test_clusters

def evaluate_dbscan(X_train, X_test, y_train, y_test, eps, min_samples):
    dbscan = fit_dbscan(X_train, eps, min_samples)
    train_clusters = dbscan.labels_
    test_clusters = test_dbscan(dbscan, X_train, X_test)

    train_clustering_scores = score_clustering(X_train, y_train, train_clusters)
    test_clustering_scores = score_clustering(X_test, y_test, test_clusters)

    train_classification_report = clustering_classification_report(train_clusters, y_train)
    test_classification_report = clustering_classification_report(test_clusters, y_test)

    clustering_metrics = format_clustering_metrics(train_clustering_scores, test_clustering_scores, eps=eps, min_samples=min_samples)
    classification_metrics = format_classification_metrics(train_classification_report, test_classification_report, eps=eps, min_samples=min_samples)
    return clustering_metrics, classification_metrics

def score_dbscan(X_train, X_test, y_train, y_test, eps=None, min_samples=None):
    if eps is None:
        eps = [0.2, 0.3, 0.4]
    if min_samples is None:
        min_samples = [5, 10]
    clustering_metrics_list = []
    classification_metrics_list = []
    for e in eps:
        for ms in min_samples:
            try:
                clustering_scores, classification_scores = evaluate_dbscan(X_train, X_test, y_train, y_test, e, ms)
                clustering_metrics_list.append(clustering_scores)
                classification_metrics_list.append(classification_scores)
            except ValueError:
                print(f'Fitting DBSCAN with eps={e} and min_samples={ms} leads to a single cluster. Skipping...')
    return clustering_metrics_list, classification_metrics_list

def fit_dbscan(X_train, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X_train)
    return dbscan

####################################################################
# Hyperparameter Tuning Utils                                      #
####################################################################

def evaluate_clustering_assignments(X, labels):
    """Compute Silhouette Score and Davies-Bouldin Index for clustering."""
    if len(set(labels)) > 1:  # Ensure more than one cluster
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        return silhouette, davies_bouldin
    else:
        return -1, float('inf')  # Invalid clustering

def tune_kmeans_dbscan(X, method, params, scoring='combined'):
    """Evaluate KMeans or DBSCAN with Silhouette and Davies-Bouldin Index."""
    best_score = float('-inf')
    best_params = None

    for param in params:
        if method == 'kmeans':
            kmeans = KMeans(**param, random_state=42)
            labels = kmeans.fit_predict(X)
        elif method == 'dbscan':
            dbscan = DBSCAN(**param)
            labels = dbscan.fit_predict(X)
        

        # Combine the two metrics (normalize DBI by its range for simplicity)
        if scoring == 'combined':
            silhouette, davies_bouldin = evaluate_clustering_assignments(X, labels)
            silhouette = (silhouette + 1) / 2
            combined_score = silhouette - davies_bouldin
            score = combined_score
        elif scoring == 'silhouette':
            score = silhouette_score(X, labels)
        elif scoring == 'davies_bouldin':
            # sign to make it maximization problem
            score = -davies_bouldin_score(X, labels)


        if score > best_score:
            best_score = score
            best_params = param

    print(f"Best Params: {best_params}, Best Combined Score: {best_score:.4f}")
    return best_params, best_score