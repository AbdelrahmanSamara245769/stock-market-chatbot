"""
Data loading utilities for the stock_chatbot project.

This module provides functions to load data from the project database for clustering
and classification tasks.

Functions:
- load_clustering_data: Loads clustering feature data from the database.
- load_classification_data: Loads classification feature data from the database.
"""

import pandas as pd
from sqlalchemy import create_engine

from .config import DB_PARAMS


def load_clustering_data():
    """
    Loads clustering feature data from the database.

    Returns:
        pd.DataFrame: DataFrame containing clustering features, indexed by date.
    """
    db_params = DB_PARAMS
    
    db_url = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
    
    engine = create_engine(db_url, echo=False)
    query = "SELECT * FROM stock_model_features;"
    df_clustering = pd.read_sql_query(query, engine, parse_dates=True, index_col="date_value")
    
    return df_clustering

def load_classification_data():
    """
    Loads classification feature data from the database.

    Returns:
        pd.DataFrame: DataFrame containing classification features.
    """
    db_params = DB_PARAMS
    
    db_url = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
    
    engine = create_engine(db_url, echo=False)
    query = "SELECT * FROM features;"
    df_classification = pd.read_sql_query(query, engine)
    
    return df_classification