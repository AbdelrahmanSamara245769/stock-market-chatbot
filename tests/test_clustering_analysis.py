import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from stock_chatbot.clustering_analysis import (
    assign_clusters, analyse_clusters, assign_cluster_categories,
    label_cluster_categories, get_safest_companies
)

def test_assign_clusters():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    labels = [0, 1]
    result = assign_clusters(df, labels)
    assert "cluster" in result.columns
    assert list(result["cluster"]) == labels

def test_analyse_clusters():
    df = pd.DataFrame({
        "value": [10, 20, 30],
        "cluster": [0, 0, 1]
    })
    result = analyse_clusters(df)
    assert isinstance(result, pd.DataFrame)
    assert 0 in result.index
    assert 1 in result.index

def test_assign_cluster_categories():
    df = pd.DataFrame({
        "return_1d_mean": [0.1, 0.05, -0.1],
        "return_1d_std": [0.01, 0.02, 0.03],
        "intraday_volatility_mean": [0.1, 0.2, 0.3],
        "volume_std": [0.01, 0.02, 0.03],
        "return_6m": [0.5, 0.3, -0.2]
    }, index=[0, 1, 2])
    categories = assign_cluster_categories(df)
    assert isinstance(categories, dict)
    assert set(categories.values()) == {'safest', 'solid', 'avoid'}

def test_label_cluster_categories():
    df = pd.DataFrame({"cluster": [0, 1, 2]}, index=["A", "B", "C"])
    mapping = {0: "safest", 1: "solid", 2: "avoid"}
    result = label_cluster_categories(df, mapping)
    assert "category" in result.columns
    assert result.loc["A", "category"] == "safest"

def test_get_safest_companies():
    df = pd.DataFrame({"category": ["safest", "avoid", "solid"]}, index=["X", "Y", "Z"])
    result = get_safest_companies(df)
    assert result == ["X"]
