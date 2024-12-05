from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from spectralnet import SpectralNet
from scipy.stats import mode


def spectral_net(X_train, X_test, max_clusters=10, random_state=42):
    torch.manual_seed(random_state)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    best_score = -1
    best_k = None
    best_model = None

    X_train_tensor = torch.from_numpy(X_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()

    # Hyperparameter tuning
    for n_clusters in range(2, max_clusters + 1):
        model = SpectralNet(n_clusters=n_clusters, spectral_hiddens=[128, 64, 16, n_clusters])

        # Train the SpectralNet model on training data
        model.fit(X_train_tensor)

        # Predict cluster labels for training data
        with torch.no_grad():
            train_cluster_labels = model.predict(X_train_tensor)

        score = silhouette_score(X_train, train_cluster_labels)
        # print(f"n_clusters: {n_clusters} with silhouette score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = n_clusters
            best_model = model

    print(f"Best K: {best_k} with silhouette score: {best_score:.4f}")

    # Predict cluster labels for training data
    with torch.no_grad():
        train_cluster_labels = best_model.predict(X_train_tensor)

    # Predict cluster labels for test data
    with torch.no_grad():
        test_cluster_labels = best_model.predict(X_test_tensor)

    # Convert cluster labels to one-hot encoded format
    train_clusters = pd.get_dummies(train_cluster_labels, prefix="Cluster").astype(int)
    test_clusters = pd.get_dummies(test_cluster_labels, prefix="Cluster").astype(int)

    # concatenate the cluster labels to the original data
    # X_train = pd.concat([pd.DataFrame(X_train), train_clusters], axis=1)
    # X_test = pd.concat([pd.DataFrame(X_test), test_clusters], axis=1)

    return train_clusters, test_clusters

