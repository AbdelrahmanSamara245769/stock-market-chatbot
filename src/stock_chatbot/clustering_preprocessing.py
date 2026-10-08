"""
Clustering preprocessing utilities for the stock_chatbot project.

This module provides functions for feature engineering, outlier handling, and scaling
to prepare data for clustering analysis.

Functions:
- engineer_features: Aggregates and engineers features for clustering from raw data.
- handle_outliers: Removes outliers from the engineered features using the IQR method.
- scale_features: Scales features using standard scaling.
- preprocess_data: Runs the full preprocessing pipeline and returns all intermediate results.
"""

import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import setup_logger

#definign logger
logger = setup_logger()

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates and engineers features for clustering from raw data.

    Args:
        df (pd.DataFrame): Raw input DataFrame with columns including 'ticker', 'close_value',
            'open_value', 'high_value', 'low_value', 'volume', 'return_1d'.

    Returns:
        pd.DataFrame: DataFrame with engineered features per ticker.
    """
    if df.index.dtype == 'object':
        df.index = pd.to_datetime(df.index)
    
    df = df.sort_index(ascending=True)
    cutoff_date = df.index.max() - pd.DateOffset(months=6)
    df_6m = df[df.index >= cutoff_date].copy()
    df_6m = df_6m[['ticker', 'close_value', 'open_value', 'high_value', 'low_value', 'volume', 'return_1d']]

    df_6m['intraday_volatility'] = (df_6m['high_value'] - df_6m['low_value']) / df_6m['open_value']
    df['return_6m'] = df.groupby('ticker')['close_value'].transform(lambda x: (x - x.shift(126)) / x.shift(126))
    return_6m = df.groupby('ticker')['return_6m'].last().reset_index()

    summary = df_6m.groupby('ticker').agg({
        'return_1d': ['mean', 'std'],
        'intraday_volatility': 'mean',
        'volume': ['mean', 'std']
    }).reset_index()

    summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in summary.columns]
    summary = summary.rename(columns={'ticker_': 'ticker'})
    summary = summary.merge(return_6m, on='ticker', how='left')
    summary.set_index('ticker', inplace=True)
    return summary

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes outliers from the engineered features using the IQR method.

    Args:
        df (pd.DataFrame): DataFrame with engineered features.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """

    Q1 = df[['return_1d_mean', 'return_1d_std', 'intraday_volatility_mean',
                  'volume_mean', 'volume_std', 'return_6m']].quantile(0.25)
    Q3 = df[['return_1d_mean', 'return_1d_std', 'intraday_volatility_mean',
                  'volume_mean', 'volume_std', 'return_6m']].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    filtered_df = df[
        (df['return_1d_mean'] >= lower_bound['return_1d_mean']) &
        (df['return_1d_mean'] <= upper_bound['return_1d_mean']) &
        (df['return_1d_std'] >= lower_bound['return_1d_std']) &
        (df['return_1d_std'] <= upper_bound['return_1d_std']) &
        (df['intraday_volatility_mean'] >= lower_bound['intraday_volatility_mean']) &
        (df['intraday_volatility_mean'] <= upper_bound['intraday_volatility_mean']) &
        (df['volume_mean'] >= lower_bound['volume_mean']) &
        (df['volume_mean'] <= upper_bound['volume_mean']) &
        (df['volume_std'] >= lower_bound['volume_std']) &
        (df['volume_std'] <= upper_bound['volume_std']) &
        (df['return_6m'] >= lower_bound['return_6m']) &
        (df['return_6m'] <= upper_bound['return_6m'])
        ]
    logger.info(f"Outliers removed: {len(df) - len(filtered_df)}")
    return filtered_df

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales features using standard scaling.

    Args:
        df (pd.DataFrame): DataFrame with features to scale.

    Returns:
        np.ndarray: Scaled feature values.
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    return scaled_features

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full preprocessing pipeline:
    1. Feature engineering
    2. Outlier removal
    3. Feature scaling

    Args:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        summary_df (pd.DataFrame): Engineered features before outlier removal.
        filtered_df (pd.DataFrame): Cleaned features with outliers removed.
        scaled_features (np.ndarray): Scaled values of filtered_df.
    """
    summary_df = engineer_features(df)
    filtered_df = handle_outliers(summary_df)
    scaled_features = scale_features(filtered_df)
    return summary_df, filtered_df, scaled_features