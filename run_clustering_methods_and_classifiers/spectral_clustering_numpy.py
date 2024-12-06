from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, diags
from sklearn.metrics import pairwise_distances_argmin



def rbf_kernel_manual_sparse(X, gamma=1.0, n_neighbors=1000):
    """
    Compute a sparse RBF kernel using nearest neighbors.
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Compute sparse RBF kernel
    row_indices = np.repeat(np.arange(X.shape[0]), n_neighbors)
    col_indices = indices.flatten()
    exp_values = np.exp(-gamma * distances.flatten()**2)

    affinity_matrix = csr_matrix((exp_values, (row_indices, col_indices)), shape=(X.shape[0], X.shape[0]))
    return affinity_matrix


def construct_laplacian(affinity_matrix):
    """
    Construct the normalized graph Laplacian.
    Supports both dense and sparse affinity matrices.
    """
    # Ensure affinity_matrix is sparse
    if not isinstance(affinity_matrix, csr_matrix):
        affinity_matrix = csr_matrix(affinity_matrix)

    # Compute degree values
    degree_values = np.array(affinity_matrix.sum(axis=1)).flatten()
    degree_values[degree_values == 0] = 1e-10  # Avoid division by zero

    # Create D^(-1/2) as a sparse diagonal matrix
    d_inv_sqrt = 1.0 / np.sqrt(degree_values)
    d_inv_sqrt_sparse = diags(d_inv_sqrt)

    # Compute the normalized Laplacian
    laplacian = diags([1.0], [0], shape=affinity_matrix.shape) - d_inv_sqrt_sparse @ affinity_matrix @ d_inv_sqrt_sparse

    return laplacian


def kmeans_pp_init(X, n_clusters, seed=42):
    np.random.seed(seed)
    n_samples, n_features = X.shape
    centroids = []

    # Step 1: Randomly select the first centroid
    first_centroid_idx = np.random.randint(0, n_samples)
    centroids.append(X[first_centroid_idx])

    # Step 2: Select remaining centroids
    for _ in range(1, n_clusters):
        # Compute distances from each point to the nearest centroid
        sq_norms_X = np.sum(X ** 2, axis=1, keepdims=True)  # Shape: (n_samples, 1)
        sq_norms_centroids = np.sum(np.array(centroids) ** 2, axis=1, keepdims=True).T  # Shape: (1, len(centroids))
        distances = sq_norms_X + sq_norms_centroids - 2 * np.dot(X, np.array(centroids).T)  # Shape: (n_samples, len(centroids))
        distances = np.sqrt(np.maximum(distances, 0))  # Ensure non-negativity

        # For each point, find the minimum distance to any centroid
        min_distances = np.min(distances, axis=1)

        # Compute the probability distribution for the next centroid
        probabilities = min_distances ** 2 / np.sum(min_distances ** 2)

        # Randomly select the next centroid based on the probabilities
        next_centroid_idx = np.random.choice(n_samples, p=probabilities)
        centroids.append(X[next_centroid_idx])

    return np.array(centroids)


def kmeans_manual(X, n_clusters, max_iter=500, tol=1e-5, seed=42):
    np.random.seed(seed)
    centroids = kmeans_pp_init(X, n_clusters, seed)
    for _ in range(max_iter):
        sq_norms_X = np.sum(X ** 2, axis=1, keepdims=True)
        sq_norms_centroids = np.sum(centroids ** 2, axis=1, keepdims=True).T
        distances = sq_norms_X + sq_norms_centroids - 2 * np.dot(X, centroids.T)
        distances = np.sqrt(np.maximum(distances, 0))
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else X[np.random.randint(0, X.shape[0])]
            for i in range(n_clusters)
        ])
        if np.allclose(centroids, new_centroids, atol=tol):
            break
        centroids = new_centroids
    return labels, centroids


def spectral_clustering_numpy(X, n_clusters, gamma=1.0, seed=42):
    np.random.seed(seed)
    affinity_matrix = rbf_kernel_manual_sparse(X, gamma=gamma)
    laplacian = construct_laplacian(affinity_matrix)
    eigvals, eigvecs = eigsh(laplacian, k=n_clusters, which='SM')  # 'SM' = Smallest Magnitude
    # Select eigenvectors corresponding to the smallest n_clusters eigenvalues
    eigvecs_subset = eigvecs[:, np.argsort(eigvals)[:n_clusters]]
    normalized_eigvecs = eigvecs_subset / np.linalg.norm(eigvecs_subset, axis=1, keepdims=True)
    labels, _ = kmeans_manual(normalized_eigvecs, n_clusters=n_clusters, seed=seed)
    return labels


def tune_spectral(X_train, X_test, max_clusters=10, random_state=42):

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    best_score = -1
    best_k = None

    # Hyperparameter tuning
    for n_clusters in range(2, max_clusters + 1):
        cluster_labels = spectral_clustering_numpy(X_train, n_clusters, gamma=1.0, seed=random_state)
        score = silhouette_score(X_train, cluster_labels)
        print(f"K: {n_clusters}, Silhouette score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = n_clusters


    print(f"Best K: {best_k} with silhouette score: {best_score:.4f}")

    # Predict cluster labels for X_train
    train_cluster_labels = spectral_clustering_numpy(X_train, best_k, gamma=1.0, seed=random_state)
    train_centroids = np.array([X_train[train_cluster_labels == i].mean(axis=0) for i in range(best_k)])

    test_cluster_labels = pairwise_distances_argmin(X_test, train_centroids, metric='euclidean')

    # Convert cluster labels to one-hot encoded format
    train_clusters = pd.get_dummies(train_cluster_labels, prefix="Cluster").astype(int)
    test_clusters = pd.get_dummies(test_cluster_labels, prefix="Cluster").astype(int)

    return train_clusters, test_clusters

