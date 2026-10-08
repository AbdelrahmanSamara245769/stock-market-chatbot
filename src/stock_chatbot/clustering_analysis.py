"""
Clustering analysis utilities for the stock_chatbot project.

This module provides functions for analyzing clustering results, assigning cluster categories,
labeling clusters, finding representative companies, and exporting results.

Functions:
- assign_clusters: Assigns cluster labels to a DataFrame.
- analyse_clusters: Computes summary statistics for each cluster.
- assign_cluster_categories: Automatically assigns categories ('safest', 'solid', 'avoid') to clusters based on scoring.
- label_cluster_categories: Maps cluster categories to the DataFrame.
- find_centroid: Finds the company closest to the centroid of the safest cluster.
- create_result_df: Combines clustering results, saves to SQL and CSV.
- get_safest_companies: Returns a list of companies in the safest cluster.
"""

import os
import logging
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import MinMaxScaler
from .config import setup_logger

#definign logger
logger = setup_logger()

def assign_clusters(df: pd.DataFrame, labels: list):
    """
    Assigns cluster labels to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        labels (list): Cluster labels.

    Returns:
        pd.DataFrame: DataFrame with 'cluster' column assigned.
    """
    df = df.copy()
    df.loc[:, 'cluster'] = labels
    return df


def analyse_clusters(df: pd.DataFrame):
    """
    Computes mean statistics for each cluster.

    Args:
        df (pd.DataFrame): DataFrame with cluster assignments.

    Returns:
        pd.DataFrame: Cluster summary statistics.
    """
    cluster_summary = df.groupby('cluster').mean(numeric_only=True)
    return cluster_summary


def assign_cluster_categories(cluster_summary: pd.DataFrame) -> dict:
    """
    Automatically assign cluster categories: 'safe', 'solid', 'avoid'
    based on normalized features and scoring.
    Returns a dictionary: {cluster_index: category}
    """
    df = cluster_summary.copy()
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    score = (
        + df_scaled['return_1d_mean'] * 0.25
        - df_scaled['return_1d_std'] * 0.25
        - df_scaled['intraday_volatility_mean'] * 0.2
        - df_scaled['volume_std'] * 0.15
        + df_scaled['return_6m'] * 0.15
    )
    ranked_clusters = score.sort_values(ascending=False).index.tolist()

    categories = ['safest', 'solid', 'avoid']
    cluster_to_category = {cluster: cat for cluster, cat in zip(ranked_clusters, categories)}
    return cluster_to_category


def label_cluster_categories(df: pd.DataFrame, cluster_to_category: dict) -> pd.DataFrame:
    """
    Maps cluster categories to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with cluster assignments.
        cluster_to_category (dict): Mapping from cluster index to category.

    Returns:
        pd.DataFrame: DataFrame with 'category' column.
    """
    df = df.copy()
    df['category'] = df['cluster'].map(cluster_to_category)
    return df


def find_centroid(filtered_df: pd.DataFrame, scaled_features: pd.DataFrame, cluster_to_category: 
dict, model):
    """
    Finds the company closest to the centroid of the safest cluster.

    Args:
        filtered_df (pd.DataFrame): DataFrame with cluster assignments.
        scaled_features (pd.DataFrame): Scaled feature DataFrame.
        cluster_to_category (dict): Mapping from cluster index to category.
        model: Fitted clustering model.

    Returns:
        str: Ticker symbol of the representative company.
    """
    centroids = model.cluster_centers_
    safe_centroid_idx = [k for k, v in cluster_to_category.items() if v == 'safest'][0]
    safe_centroid = centroids[safe_centroid_idx]

    cluster_1_data = scaled_features[filtered_df['cluster'] == safe_centroid_idx]
    cluster_1_tickers = filtered_df[filtered_df['cluster'] == safe_centroid_idx].index.values

    closest_idx, _ = pairwise_distances_argmin_min([safe_centroid], cluster_1_data)
    representative_ticker = cluster_1_tickers[closest_idx]
    logger.info(f'Closest company to centroid of the safest cluster: {representative_ticker}')
    return representative_ticker


def create_result_df(summary_df: pd.DataFrame, filtered_df: pd.DataFrame, cluster_to_category: dict, engine):
    """
    Combines clustering results, saves to SQL and CSV.

    Args:
        summary_df (pd.DataFrame): Cluster summary DataFrame.
        filtered_df (pd.DataFrame): DataFrame with cluster assignments.
        cluster_to_category (dict): Mapping from cluster index to category.
        engine: SQLAlchemy engine.

    Returns:
        pd.DataFrame: Result DataFrame with cluster and category.
    """
    result = summary_df.copy()
    result['cluster'] = result.index.map(filtered_df['cluster'])
    result['category'] = result['cluster'].map(cluster_to_category)
    result['category'] = result['category'].fillna('outlier')
    result.reset_index(inplace=True)

    result.to_sql(name='clustering_results', con=engine, if_exists='replace', index=False)
    logger.info("Clustering results SQL table created successfully.")

    os.makedirs("clustering_output", exist_ok=True)
    result.to_csv('clustering_output/clustering_results.csv', index=False)
    logger.info("Clustering results CSV table saved successfully.")
    return result


def get_safest_companies(df: pd.DataFrame) -> list:
    """
    Get list of companies from the safest cluster
    
    Args:
        df: DataFrame containing cluster assignments and categories
        
    Returns:
        list: List of ticker symbols for companies in the safest cluster
    """
    return df[df['category'] == 'safest'].index.tolist()
