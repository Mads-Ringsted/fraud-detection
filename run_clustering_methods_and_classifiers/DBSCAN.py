from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def dbscan(X_train, X_test, eps_list=None, min_samples_list=None):
    best_score = -1
    best_params = None
    best_model = None
    best_labels = None

    if eps_list is None:
        eps_list = [0.1, 0.15, 0.2, 0.25, 0.3]

    if min_samples_list is None:
        min_samples_list = [5, 10]

    # Hyperparameter tuning
    for eps in eps_list:
        for min_samples in min_samples_list:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_train)
            
            # Only calculate silhouette score if more than one cluster is found
            if len(set(labels)) > 1:
                score = silhouette_score(X_train, labels)
                if score > best_score:
                    best_score = score
                    best_params = {'eps': eps, 'min_samples': min_samples}
                    best_model = model
                    best_labels = labels
    
    print(f"Best Params: {best_params} with silhouette score: {best_score:.4f}")

    # Predict cluster labels for X_train
    train_labels = best_labels
    train_clusters = pd.get_dummies(train_labels, prefix="Cluster").astype(int)
    # Predict cluster labels for X_test
    test_labels = test_dbscan(best_model, X_train, X_test)
    test_clusters = pd.get_dummies(test_labels, prefix="Cluster").astype(int)

    # Concatenate the cluster labels to the original data
    X_train = pd.concat([pd.DataFrame(X_train), train_clusters], axis=1)
    X_test = pd.concat([pd.DataFrame(X_test), test_clusters], axis=1)

    return X_train, X_test

def test_dbscan(dbscan, X_train, X_test):
    core_samples_mask = dbscan.core_sample_indices_
    core_points = X_train[core_samples_mask]

    nn = NearestNeighbors(n_neighbors=1).fit(core_points)
    distances, indices = nn.kneighbors(X_test)

    test_clusters = np.array([
        dbscan.labels_[core_samples_mask[i]] if distances[j] < dbscan.eps else -1
        for j, i in enumerate(indices.flatten())
    ])
    return test_clusters


if __name__ == "__main__":
    data = pd.read_csv('creditcard.csv')
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

    train_clusters, test_clusters = tune_dbscan(X_train, X_test)

    print(test_clusters.head())