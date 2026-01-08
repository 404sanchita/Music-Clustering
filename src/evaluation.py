import numpy as np
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)

def calculate_all_metrics(data, predicted_labels, true_labels=None):
    """Calculate all clustering evaluation metrics"""
    metrics = {}
    
    # Internal metrics (don't require true labels)
    try:
        metrics['silhouette_score'] = silhouette_score(data, predicted_labels)
    except:
        metrics['silhouette_score'] = np.nan
    
    try:
        metrics['calinski_harabasz_index'] = calinski_harabasz_score(data, predicted_labels)
    except:
        metrics['calinski_harabasz_index'] = np.nan
    
    try:
        metrics['davies_bouldin_index'] = davies_bouldin_score(data, predicted_labels)
    except:
        metrics['davies_bouldin_index'] = np.nan
    
    # External metrics (require true labels)
    if true_labels is not None:
        try:
            metrics['adjusted_rand_index'] = adjusted_rand_score(true_labels, predicted_labels)
        except:
            metrics['adjusted_rand_index'] = np.nan
        
        try:
            metrics['nmi'] = normalized_mutual_info_score(true_labels, predicted_labels)
        except:
            metrics['nmi'] = np.nan
        
        try:
            metrics['cluster_purity'] = calculate_cluster_purity(true_labels, predicted_labels)
        except:
            metrics['cluster_purity'] = np.nan
    else:
        metrics['adjusted_rand_index'] = np.nan
        metrics['nmi'] = np.nan
        metrics['cluster_purity'] = np.nan
    
    return metrics

def calculate_cluster_purity(true_labels, predicted_labels):
    """Calculate cluster purity"""
    # Count the most common class in each cluster
    n_samples = len(true_labels)
    clusters = np.unique(predicted_labels)
    purity = 0.0
    
    for cluster_id in clusters:
        if cluster_id == -1:  # Skip noise points in DBSCAN
            continue
        cluster_mask = predicted_labels == cluster_id
        if np.sum(cluster_mask) == 0:
            continue
        cluster_true_labels = true_labels[cluster_mask]
        most_common_count = np.bincount(cluster_true_labels).max()
        purity += most_common_count
    
    return purity / n_samples

def print_metrics(metrics, title="Clustering Metrics"):
    """Print metrics in a formatted way"""
    print(f"\n{title}")
    print("-" * 50)
    print(f"Silhouette Score:        {metrics['silhouette_score']:.4f}")
    print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_index']:.4f}")
    print(f"Davies-Bouldin Index:    {metrics['davies_bouldin_index']:.4f}")
    if not np.isnan(metrics['adjusted_rand_index']):
        print(f"Adjusted Rand Index:     {metrics['adjusted_rand_index']:.4f}")
        print(f"Normalized MI:           {metrics['nmi']:.4f}")
        print(f"Cluster Purity:          {metrics['cluster_purity']:.4f}")

