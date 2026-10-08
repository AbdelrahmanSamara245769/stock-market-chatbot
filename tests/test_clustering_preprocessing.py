import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from stock_chatbot.clustering_preprocessing import (
    engineer_features,
    handle_outliers,
    scale_features,
    preprocess_data
)

def create_sample_data():
    """
    Create a simple multi-ticker financial dataset with datetime index.
    Covers a span of >6 months to allow feature engineering.
    """
    np.random.seed(0)
    dates = pd.date_range(end=datetime.today(), periods=200)
    data = []

    for ticker in ['AAA', 'BBB']:
        for date in dates:
            open_value = np.random.uniform(100, 200)
            high_value = open_value + np.random.uniform(0, 10)
            low_value = open_value - np.random.uniform(0, 10)
            close_value = np.random.uniform(low_value, high_value)
            volume = np.random.randint(1000, 5000)
            return_1d = np.random.normal(0, 0.01)
            data.append([date, ticker, close_value, open_value, high_value, low_value, volume, return_1d])

    df = pd.DataFrame(data, columns=[
        'date', 'ticker', 'close_value', 'open_value', 'high_value',
        'low_value', 'volume', 'return_1d'
    ])
    df.set_index('date', inplace=True)
    return df

def test_engineer_features():
    df = create_sample_data()
    summary_df = engineer_features(df)

    # Check structure
    assert isinstance(summary_df, pd.DataFrame)
    expected_cols = {
        'return_1d_mean', 'return_1d_std', 'intraday_volatility_mean',
        'volume_mean', 'volume_std', 'return_6m'
    }
    assert expected_cols.issubset(summary_df.columns)
    assert not summary_df.isnull().values.any()

def test_handle_outliers():
    df = create_sample_data()
    summary_df = engineer_features(df)
    filtered_df = handle_outliers(summary_df)

    # Should return a subset of original DataFrame
    assert isinstance(filtered_df, pd.DataFrame)
    assert len(filtered_df) <= len(summary_df)

    # Ensure no NaNs
    assert not filtered_df.isnull().values.any()

def test_scale_features():
    df = create_sample_data()
    summary_df = engineer_features(df)
    filtered_df = handle_outliers(summary_df)
    scaled = scale_features(filtered_df)

    assert isinstance(scaled, np.ndarray)
    assert scaled.shape[0] == filtered_df.shape[0]
    assert scaled.shape[1] == filtered_df.shape[1]

    # Mean ~0 and std ~1 for each feature (numerical check)
    np.testing.assert_allclose(scaled.mean(axis=0), np.zeros(scaled.shape[1]), atol=1e-1)
    np.testing.assert_allclose(scaled.std(axis=0), np.ones(scaled.shape[1]), atol=1e-1)

def test_preprocess_data():
    df = create_sample_data()
    summary_df, filtered_df, scaled = preprocess_data(df)

    assert isinstance(summary_df, pd.DataFrame)
    assert isinstance(filtered_df, pd.DataFrame)
    assert isinstance(scaled, np.ndarray)
    assert filtered_df.shape[0] == scaled.shape[0]
    assert filtered_df.shape[1] == scaled.shape[1]
