"""
Model utilities for the stock_chatbot project.

This module provides functions for loading classification models and encoders,
scoring new data, and summarizing predictions.

Functions:
- load_model_and_encoder: Loads a trained model and label encoder from disk.
- load_models: Loads the default model and encoder for the project.
- score_new_data: Scores new data using the classification model and returns predictions.
- get_predictions_summary: Returns a summary DataFrame of predictions.
"""

import logging
import pandas as pd
import joblib
from pathlib import Path
from pandas.tseries.offsets import MonthEnd

from .data_loader import load_classification_data
from .config import CLASSIFICATION_FEATURES
from .config import setup_logger
from .classification_preprocessing import preprocess_classification_features

#definign logger
logger = setup_logger()


def load_model_and_encoder(model_path: str, encoder_path: str):
    """
    Loads a trained model and label encoder from disk.

    Args:
        model_path (str): Path to the trained model file.
        encoder_path (str): Path to the label encoder file.

    Returns:
        model: Loaded model object.
        encoder: Loaded label encoder object.
    """
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder


def load_models():
    """
    Loads the default model and encoder for the project.

    Returns:
        model: Loaded model object.
        encoder: Loaded label encoder object.
    """
    base_path = Path(__file__).parent.parent.parent
    xgb_path = base_path / 'models' / 'xgb_ctsh_model.pkl'
    encoder_path = base_path / 'models' / 'label_encoder_ctsh.pkl'
    return load_model_and_encoder(xgb_path, encoder_path)


def score_new_data(model_path: str, encoder_path: str, safe_companies: list = None):
    """
    Scores new data using the classification model and returns predictions.

    Args:
        model_path (str): Path to the trained model file.
        encoder_path (str): Path to the label encoder file.
        safe_companies (list, optional): List of company prefixes to filter predictions.

    Returns:
        pd.DataFrame: DataFrame with predictions for the target date.
    """
    df_to_score = load_classification_data()
    df_to_score["date_value"] = pd.to_datetime(df_to_score["date_value"])

    # --- Apply preprocessing here ---
    classification_df = preprocess_classification_features(df_to_score)

    # Find the latest date in the processed data
    latest_date = classification_df["date_value"].max()
    logger.info(f"Latest date in processed data: {latest_date}")

    # Calculate the target date (6 months from latest date)
    target_date = latest_date + pd.DateOffset(months=6)
    target_date = target_date + MonthEnd(0)
    logger.info(f"Predicting for target date: {target_date}")

    # Create future data points using the latest data
    latest_data = classification_df[classification_df["date_value"] == latest_date].copy()
    latest_data["date_value"] = target_date

    # Filter for safe companies
    if safe_companies is not None:
        latest_data = latest_data[latest_data['company_prefix'].isin(safe_companies)]
        logger.info(f"Filtering predictions for {len(safe_companies)} safe companies")

    if len(latest_data) == 0:
        logger.info("No data available for prediction")
        return pd.DataFrame()

    X_new = latest_data[CLASSIFICATION_FEATURES]
    model, label_encoder = load_model_and_encoder(model_path, encoder_path)

    y_pred_enc = model.predict(X_new)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    latest_data["predicted_price_label"] = y_pred

    logger.info(f"Made predictions for {len(latest_data)} companies at {target_date}")
    return latest_data


def get_predictions_summary(df):
    """
    Get a summary of the predictions.

    Args:
        df (pd.DataFrame): DataFrame with prediction results.

    Returns:
        pd.DataFrame: DataFrame with company, date, and predicted label.
    """
    """
    Get a summary of the predictions
    """
    return df[["company_prefix", "date_value", "predicted_price_label"]]


