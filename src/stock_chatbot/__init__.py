"""
stock_chatbot package

This package provides tools for data loading, preprocessing, clustering, and classification analysis
for the stock_chatbot project. It exposes functions for scoring new data, summarizing predictions,
loading datasets, performing clustering, and analyzing cluster results.

Modules imported:
- model: Functions for scoring and summarizing predictions.
- data_loader: Functions for loading clustering and classification data.
- config: Configuration constants for features.
- clustering_preprocessing: Data preprocessing for clustering.
- clustering_model: Clustering model execution.
- clustering_analysis: Cluster assignment, analysis, labeling, and centroid finding.

Exports:
- score_new_data, get_predictions_summary
- load_clustering_data, load_classification_data
- CLASSIFICATION_FEATURES
- preprocess_data, perform_clustering
- assign_clusters, analyse_clusters, assign_cluster_categories, label_cluster_categories, find_centroid, get_safest_companies
"""

from .model import (
    score_new_data,
    get_predictions_summary
)

from .data_loader import (
    load_clustering_data,
    load_classification_data
)

from .config import CLASSIFICATION_FEATURES

from .clustering_preprocessing import preprocess_data
from .clustering_model import perform_clustering
from .clustering_analysis import (
    assign_clusters,
    analyse_clusters,
    assign_cluster_categories,
    label_cluster_categories,
    find_centroid,
    get_safest_companies
)

__all__ = [
    'find_representative_company',
    'get_safest_companies',
    'analyze_clusters',
    
    'score_new_data',
    'get_predictions_summary',
    
    'load_clustering_data',
    'load_classification_data',
    
    'CLUSTERING_FEATURES',
    'CLASSIFICATION_FEATURES'
]
