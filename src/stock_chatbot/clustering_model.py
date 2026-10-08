"""
Clustering model utilities for the stock_chatbot project.

This module provides the function to perform K-Means clustering on scaled feature data.

Functions:
- perform_clustering: Runs K-Means clustering and returns the fitted model and cluster labels.
"""

import pandas as pd
from sklearn.cluster import KMeans

def perform_clustering(scaled_features: pd.DataFrame):
    """
    Run K-Means clustering on the data.

    Args:
        scaled_features (pd.DataFrame): DataFrame of scaled features for clustering.

    Returns:
        model (KMeans): Fitted KMeans clustering model.
        labels (np.ndarray): Cluster labels for each sample.
    """
    model = KMeans(n_clusters=3, random_state=12)
    labels = model.fit_predict(scaled_features)
    return model, labels