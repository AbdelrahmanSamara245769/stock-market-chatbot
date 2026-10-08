import pandas as pd
from unittest.mock import patch
from stock_chatbot.data_loader import load_clustering_data, load_classification_data

@patch("stock_chatbot.data_loader.create_engine")
@patch("pandas.read_sql_query")
def test_load_clustering_data(mock_read_sql, mock_engine):
    mock_read_sql.return_value = pd.DataFrame({"col": [1, 2]})
    df = load_clustering_data()
    assert isinstance(df, pd.DataFrame)

@patch("stock_chatbot.data_loader.create_engine")
@patch("pandas.read_sql_query")
def test_load_classification_data(mock_read_sql, mock_engine):
    mock_read_sql.return_value = pd.DataFrame({"col": [1, 2]})
    df = load_classification_data()
    assert isinstance(df, pd.DataFrame)
