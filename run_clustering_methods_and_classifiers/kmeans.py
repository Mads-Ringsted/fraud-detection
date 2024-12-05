from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def tune_kmeans(X_train, X_test=None, max_clusters=10, random_state=42):
    best_score = -1
    best_k = None
    best_model = None

    # Hyperparameter tuning
    for n_clusters in range(2, max_clusters + 1):
        model = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = model.fit_predict(X_train)
        score = silhouette_score(X_train, cluster_labels)
        
        if score > best_score:
            best_score = score
            best_k = n_clusters
            best_model = model
    
    print(f"Best K: {best_k} with silhouette score: {best_score:.4f}")

    # Predict cluster labels for X_train
    X_train_labels = best_model.predict(X_train)
    train_clusters = pd.get_dummies(X_train_labels, prefix="Cluster").astype(int)
    
    if X_test is not None:
        # Predict cluster labels for X_test
        X_test_labels = best_model.predict(X_test)
        test_clusters = pd.get_dummies(X_test_labels, prefix="Cluster").astype(int)
        return train_clusters, test_clusters

    return train_clusters


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

    train_clusters, test_clusters = tune_kmeans(X_train, X_test)

    print(train_clusters)
