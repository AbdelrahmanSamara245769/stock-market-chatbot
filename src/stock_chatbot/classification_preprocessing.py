"""
Feature engineering and preprocessing for classification tasks.

This module provides functions to preprocess and engineer features for classification models
using financial time series data. It includes rolling statistics, technical indicators,
outlier capping, and feature imputation for a single ticker DataFrame.

Functions:
- preprocess_classification_features: Generates engineered features and handles missing values/outliers.
"""

import pandas as pd
import numpy as np

def preprocess_classification_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses and engineers features for classification on a single ticker DataFrame.

    Assumes df is sorted by date and contains all required columns:
    - 'close_value': Closing price of the asset.
    - 'volume': Trading volume.

    Feature engineering includes:
    - Moving averages (MA), volatility, returns, momentum, rate of change (ROC)
    - Bollinger Bands, RSI-14, MA crossovers, Sharpe ratio, volume change
    - Outlier capping for selected features using IQR
    - Imputation of missing rolling features

    Returns:
        pd.DataFrame: DataFrame with engineered features and capped outliers.
    """
    df = df.copy()
    # Feature engineering
    df['ma_126'] = df['close_value'].rolling(window=126).mean()
    df['volatility_126'] = df['close_value'].rolling(window=126).std()
    df['return_1d'] = df['close_value'].pct_change(periods=1) * 100
    df['momentum_126'] = df['close_value'] - df['close_value'].shift(126)
    df['roc_126'] = df['close_value'].pct_change(periods=126) * 100

    ma_20 = df['close_value'].rolling(window=20).mean()
    std_20 = df['close_value'].rolling(window=20).std()
    df['bollinger_upper'] = ma_20 + 2 * std_20
    df['bollinger_lower'] = ma_20 - 2 * std_20
    df['bollinger_bandwidth'] = (df['bollinger_upper'] - df['bollinger_lower']) / ma_20

    # RSI-14
    delta = df['close_value'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MA crossover
    ma_50 = df['close_value'].rolling(window=50).mean()
    df['ma_crossover_20_50'] = (ma_20 > ma_50).astype(int)

    # Sharpe ratio (126d)
    mean_return_126 = df['return_1d'].rolling(window=126).mean()
    std_return_126 = df['return_1d'].rolling(window=126).std()
    df['sharpe_126'] = mean_return_126 / std_return_126

    df['volume_change_1d'] = df['volume'].pct_change(periods=1) * 100

    rolling_cols = ['ma_126', 'volatility_126', 'roc_126', 'rsi_14', 'sharpe_126']

    # Impute missing rolling features using backward then forward fill, then fill any remaining with column mean
    df[rolling_cols] = df[rolling_cols].bfill().ffill()
    df[rolling_cols] = df[rolling_cols].fillna(df[rolling_cols].mean())

    # Outlier capping for return_6m
    """
       Caps outliers in a pandas Series using the IQR method.
    """
    def cap_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return series.clip(lower, upper)

    df['rsi_14_capped'] = cap_iqr(df['rsi_14'])
    df['sharpe_126_capped'] = cap_iqr(df['sharpe_126'])
    df['volatility_126_capped'] = cap_iqr(df['volatility_126'])

    # Classification target
    def classify_6m_capped(val):
        """
        Classifies 6-month capped return into categories.
        """
        if pd.isna(val):
            return np.nan
        if val >= 30:
            return 'very_high'
        elif val >= 15:
            return 'high'
        elif val >= -5:
            return 'no_change'
        elif val >= -20:
            return 'low'
        else:
            return 'very_low'

    classification_df = df

    return classification_df