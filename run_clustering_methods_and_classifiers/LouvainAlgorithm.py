import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from community import community_louvain

# Load dataset
df = pd.read_csv('creditcard.csv')

# Separate features and target variable
X = df.drop(['Class'], axis=1)
y = df['Class']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def louvain_algorithm(X_train, X_test):
    modularity_scores = []
    k_values = range(5, 30, 5) 

    for k in k_values:
        # Construct KNN graph
        nn = NearestNeighbors(n_neighbors=k, metric='cosine')
        nn.fit(X_train)
        distances, indices = nn.kneighbors(X_train)
        
        # Build graph
        G = nx.Graph() 
        for i, neighbors in enumerate(indices):
            for j, dist in zip(neighbors, distances[i]):
                if i != j:
                    G.add_edge(i, j, weight=1 - dist)
        
        # Apply Louvain clustering
        partition = community_louvain.best_partition(G)
        modularity = community_louvain.modularity(partition, G)
        modularity_scores.append((k, modularity))

    # Find k with the highest modularity
    optimal_k = max(modularity_scores, key=lambda x: x[1])[0]
    print(f"Optimal k based on modularity: {optimal_k}")

    # Print also modularity scores for all k values
    print(modularity_scores)

    # Recompute with optimal_k
    nn = NearestNeighbors(n_neighbors=optimal_k, metric='cosine')
    nn.fit(X_train)
    distances, indices = nn.kneighbors(X_train)
    
    G = nx.Graph() 
    for i, neighbors in enumerate(indices):
        for j, dist in zip(neighbors, distances[i]):
            if i != j:
                G.add_edge(i, j, weight=1 - dist)
    
    partition = community_louvain.best_partition(G)

    # Map partition dictionary to a list aligned with training data indices
    train_clusters = np.array([partition[i] for i in range(len(X_train))])

    # Compute cluster prototypes (mean vectors of clusters)
    cluster_ids = np.unique(train_clusters)
    cluster_prototypes = {}

    for cluster_id in cluster_ids:
        cluster_members = X_train[train_clusters == cluster_id]
        cluster_prototype = cluster_members.mean(axis=0)
        cluster_prototypes[cluster_id] = cluster_prototype

    def compute_cluster_distances(X, cluster_prototypes):
        distances = []
        for x in X:
            dists = [np.linalg.norm(x - cluster_prototypes[cluster_id]) for cluster_id in cluster_prototypes]
            distances.append(dists)
        return np.array(distances)

    # Compute distances for training data
    train_cluster_distances = compute_cluster_distances(X_train, cluster_prototypes)

    # Compute distances for test data
    test_cluster_distances = compute_cluster_distances(X_test, cluster_prototypes)

    # Combine original features with cluster distance features
    X_train_with_features = np.hstack((X_train, train_cluster_distances))
    X_test_with_features = np.hstack((X_test, test_cluster_distances))

    return X_train_with_features, X_test_with_features
