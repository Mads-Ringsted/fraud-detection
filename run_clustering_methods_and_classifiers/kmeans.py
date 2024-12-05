from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def kmeans(X_train, X_test, max_clusters=10, random_state=42):

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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
    
    # Predict cluster labels for X_test
    X_test_labels = best_model.predict(X_test)
    test_clusters = pd.get_dummies(X_test_labels, prefix="Cluster").astype(int)

    # concatenate the cluster labels to the original data
    # X_train = pd.concat([pd.DataFrame(X_train), train_clusters], axis=1)
    # X_test = pd.concat([pd.DataFrame(X_test), test_clusters], axis=1)

    return train_clusters, test_clusters
