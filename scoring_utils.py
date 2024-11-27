import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, f1_score, precision_score, recall_score, silhouette_score, accuracy_score


def purity_score(X, y, cluster_labels) -> float:
    """
    Function to calculate the purity score. Which is a measure of how well a cluster contains only one class.
    It is calculated as the fraction of the dominant class in the cluster. The purity score is then adjusted based on the expected purity.
    """

    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    global_class_distribution = y.value_counts(normalize=True).to_dict() # this mihgt be wrong as we are using splits, but they are stratified so it should be okay
    # Compute metrics for each cluster, including adjusted purity
    cluster_purity = []
    cluster_weight = []

    for cluster_id in np.unique(cluster_labels):
        idx = np.where(cluster_labels == cluster_id)
        cluster_data = X[idx]
        total_in_cluster = len(cluster_data)
        
        # Compute class distribution within the cluster
        class_distribution = y.iloc[idx].value_counts(normalize=True)
        
        # Purity: Fraction of the dominant class in the cluster
        dominant_class = class_distribution.idxmax()
        purity = class_distribution[dominant_class]
        
        # Expected purity based on global distribution
        expected_purity = global_class_distribution[dominant_class]
        
        # Adjusted Purity
        if purity > expected_purity:
            adjusted_purity = (purity - expected_purity) / (1 - expected_purity)
        else:
            adjusted_purity = 0  # Set to 0 if purity is less than or equal to expected purity
        
        # Weighted purity
        cluster_purity.append(adjusted_purity)
        cluster_weight.append(total_in_cluster / len(X))
    
    # Compute the weighted average of cluster purity
    purity = np.sum(np.array(cluster_purity) * np.array(cluster_weight))
    return purity

def score_clustering(X, y, cluster_labels):
    return davies_bouldin_score(X, cluster_labels), silhouette_score(X, cluster_labels), purity_score(X, y, cluster_labels)

def clustering_classification_report(cluster_labels, class_labels, use_class_fraction=False):
    """
    Function to calculate the accuracy of a clustering algorithm. It does this by finding the most common class in each cluster and then mapping the cluster labels to the class labels.
    """
    df = pd.DataFrame({'cluster': cluster_labels, 'class': class_labels})

    # for each cluster find the class that is most common
    if use_class_fraction:
        cluster_class = df.groupby('cluster')['class'].apply(lambda x: 1 if (x.mean() > y.mean()) else 0) # here y is the global variable
    else:
        cluster_class = df.groupby('cluster')['class'].agg(lambda x:x.value_counts().index[0])

    # map the cluster labels to the class labels
    predicted_class = df['cluster'].map(cluster_class)
    # calculate the accuracy
    accuracy = accuracy_score(class_labels, predicted_class)
    f1 = f1_score(class_labels, predicted_class)
    recall = recall_score(class_labels, predicted_class)
    precision = precision_score(class_labels, predicted_class)
    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'recall': recall,
        'precision': precision
    }
    return metrics
    