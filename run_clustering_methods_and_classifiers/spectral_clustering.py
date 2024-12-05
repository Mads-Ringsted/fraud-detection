from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances_argmin
import numpy as np




def tune_spectral_clustering(X_train, X_test, max_clusters=10, random_state=42):
    np.random.seed(random_state)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    best_score = -1
    best_k = None
    best_model = None


    # Hyperparameter tuning
    for n_clusters in range(2, max_clusters + 1):
        model = SpectralClustering(n_clusters=n_clusters, affinity='rbf', gamma=1.0, assign_labels='kmeans', random_state=random_state)
        train_cluster_labels = model.fit_predict(X_train)

        score = silhouette_score(X_train, train_cluster_labels)
        print(f"n_clusters: {n_clusters} with silhouette score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = n_clusters
            best_model = model

    print(f"Best K: {best_k} with silhouette score: {best_score:.4f}")

    train_cluster_labels = best_model.fit_predict(X_train)

    # Approximate cluster centroids based on X_train clusters
    train_centroids = np.array([X_train[train_cluster_labels == i].mean(axis=0) for i in range(best_k)])
    test_cluster_labels = pairwise_distances_argmin(X_test, train_centroids, metric='euclidean')

    # Convert cluster labels to one-hot encoded format
    train_clusters = pd.get_dummies(train_cluster_labels, prefix="Cluster").astype(int)
    test_clusters = pd.get_dummies(test_cluster_labels, prefix="Cluster").astype(int)

    return train_clusters, test_clusters


if __name__ == "__main__":
    data = pd.read_csv('../creditcard.csv')
    X = data.drop(['Class', 'Amount', 'Time'], axis=1)
    y = data['Class']

    X_scale = MinMaxScaler().fit_transform(X)

    # sample data to reduce class imbalance
    non_fraud_df = X_scale[y == 0][:2000]
    fraud_df = X_scale[y == 1]

    X_sample = np.vstack([non_fraud_df, fraud_df])
    fraud_idx = np.zeros(len(X_sample))
    fraud_idx[-len(fraud_df):] = 1

    indices = np.arange(len(X_sample))
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X_sample, fraud_idx, indices, test_size=0.2, random_state=42, stratify=fraud_idx)

    train_clusters, test_clusters = tune_spectral_clustering(X_train, X_test)

    print(train_clusters)
