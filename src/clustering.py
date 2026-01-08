import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
import sklearn

def kmeans_clustering(data, n_clusters=6, random_state=42):
    """K-Means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(data)
    return labels, kmeans

def agglomerative_clustering(data, n_clusters=6, linkage='ward', metric='euclidean'):
    """Agglomerative Hierarchical Clustering"""
    # For 'ward' linkage, only 'euclidean' metric is supported.
    # Newer scikit-learn versions use 'metric' instead of 'affinity'.
    if linkage == 'ward':
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    else:
        # For other linkages, 'metric' parameter is used in newer sklearn
        # and 'affinity' in older versions.
        if sklearn.__version__ >= '1.2':
            agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric=metric)
        else:
            agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=metric)
    
    labels = agg_clustering.fit_predict(data)
    return labels, agg_clustering

def dbscan_clustering(data, eps=0.5, min_samples=5, standardize=True):
    """DBSCAN density-based clustering"""
    if standardize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    return labels, dbscan

def apply_all_clustering_algorithms(data, n_clusters=6, standardize=True):
    """Apply all clustering algorithms and return results"""
    results = {}
    
    if standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data
    
    # K-Means
    labels_kmeans, model_kmeans = kmeans_clustering(data_scaled, n_clusters=n_clusters)
    results['kmeans'] = {
        'labels': labels_kmeans,
        'model': model_kmeans,
        'n_clusters': n_clusters
    }
    
    # Agglomerative Clustering
    labels_agg, model_agg = agglomerative_clustering(data_scaled, n_clusters=n_clusters)
    results['agglomerative'] = {
        'labels': labels_agg,
        'model': model_agg,
        'n_clusters': n_clusters
    }
    
    # DBSCAN (may produce different number of clusters)
    labels_dbscan, model_dbscan = dbscan_clustering(data_scaled, eps=0.5, min_samples=5, standardize=False)
    n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    results['dbscan'] = {
        'labels': labels_dbscan,
        'model': model_dbscan,
        'n_clusters': n_clusters_dbscan
    }
    
    return results

