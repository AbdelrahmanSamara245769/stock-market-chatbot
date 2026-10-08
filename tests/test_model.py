import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from stock_chatbot import model


@patch("stock_chatbot.model.joblib.load")
def test_load_model_and_encoder(mock_joblib_load):
    mock_model = MagicMock()
    mock_encoder = MagicMock()
    mock_joblib_load.side_effect = [mock_model, mock_encoder]

    model_path = "fake_model.pkl"
    encoder_path = "fake_encoder.pkl"
    loaded_model, loaded_encoder = model.load_model_and_encoder(model_path, encoder_path)

    assert loaded_model == mock_model
    assert loaded_encoder == mock_encoder
    assert mock_joblib_load.call_count == 2


@patch("stock_chatbot.model.load_model_and_encoder")
@patch("stock_chatbot.model.preprocess_classification_features")
@patch("stock_chatbot.model.load_classification_data")
def test_score_new_data_with_predictions(mock_load_data, mock_preprocess, mock_load_models):
    # Fake raw and preprocessed data
    df_raw = pd.DataFrame({
        "date_value": ["2024-01-01", "2024-01-01"],
        "company_prefix": ["AAA", "BBB"]
    })
    df_processed = pd.DataFrame({
        "date_value": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "company_prefix": ["AAA", "BBB"],
        **{feature: [0.5, 0.7] for feature in model.CLASSIFICATION_FEATURES}
    })

    mock_load_data.return_value = df_raw
    mock_preprocess.return_value = df_processed

    mock_model = MagicMock()
    mock_encoder = MagicMock()
    mock_model.predict.return_value = [0, 1]
    mock_encoder.inverse_transform.return_value = ["increase", "decrease"]
    mock_load_models.return_value = (mock_model, mock_encoder)

    result_df = model.score_new_data("fake_model.pkl", "fake_encoder.pkl")

    assert "predicted_price_label" in result_df.columns
    assert result_df.loc[0, "predicted_price_label"] in ["increase", "decrease"]
    assert len(result_df) == 2


@patch("stock_chatbot.model.load_model_and_encoder")
@patch("stock_chatbot.model.preprocess_classification_features")
@patch("stock_chatbot.model.load_classification_data")
def test_score_new_data_with_safe_companies_filter(mock_load_data, mock_preprocess, mock_load_models):
    df_raw = pd.DataFrame({
        "date_value": ["2024-01-01", "2024-01-01"],
        "company_prefix": ["AAA", "BBB"]
    })
    df_processed = pd.DataFrame({
        "date_value": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "company_prefix": ["AAA", "BBB"],
        **{feature: [0.5, 0.7] for feature in model.CLASSIFICATION_FEATURES}
    })

    mock_load_data.return_value = df_raw
    mock_preprocess.return_value = df_processed

    mock_model = MagicMock()
    mock_encoder = MagicMock()
    mock_model.predict.return_value = [0]
    mock_encoder.inverse_transform.return_value = ["increase"]
    mock_load_models.return_value = (mock_model, mock_encoder)

    safe = ["AAA"]  # Only one company is safe
    result_df = model.score_new_data("model.pkl", "encoder.pkl", safe_companies=safe)

    assert all(result_df["company_prefix"].isin(safe))
    assert len(result_df) == 1


def test_get_predictions_summary():
    df = pd.DataFrame({
        "company_prefix": ["AAA", "BBB"],
        "date_value": ["2024-01-01", "2024-01-01"],
        "predicted_price_label": ["increase", "decrease"],
        "extra_column": [1, 2]
    })
    summary = model.get_predictions_summary(df)

    assert list(summary.columns) == ["company_prefix", "date_value", "predicted_price_label"]
    assert len(summary) == 2
