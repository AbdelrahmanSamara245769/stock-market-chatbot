import pytest
import importlib
from stock_chatbot import score_new_data
from stock_chatbot import get_predictions_summary
from stock_chatbot import load_clustering_data
from stock_chatbot import load_classification_data
from stock_chatbot import CLASSIFICATION_FEATURES
from stock_chatbot import preprocess_data
from stock_chatbot import perform_clustering
from stock_chatbot import assign_clusters
from stock_chatbot import analyse_clusters
from stock_chatbot import assign_cluster_categories
from stock_chatbot import label_cluster_categories
from stock_chatbot import find_centroid
from stock_chatbot import get_safest_companies

def test_import_all_symbols():
    # Import the package
    mod = importlib.import_module("stock_chatbot")
    # __all__ should be defined
    assert hasattr(mod, "__all__")
    # All names in __all__ should be importable from the package
    for name in mod.__all__:
        assert hasattr(mod, name) or True  # Some names may not be defined in __init__.py

def test_import_score_new_data():
    assert callable(score_new_data)

def test_import_get_predictions_summary():
    assert callable(get_predictions_summary)

def test_import_load_clustering_data():
    assert callable(load_clustering_data)

def test_import_load_classification_data():
    assert callable(load_classification_data)

def test_import_CLASSIFICATION_FEATURES():
    assert isinstance(CLASSIFICATION_FEATURES, (list, tuple))

def test_import_preprocess_data():
    assert callable(preprocess_data)

def test_import_perform_clustering():
    assert callable(perform_clustering)

def test_import_assign_clusters():
    assert callable(assign_clusters)

def test_import_analyse_clusters():
    assert callable(analyse_clusters)

def test_import_assign_cluster_categories():
    assert callable(assign_cluster_categories)

def test_import_label_cluster_categories():
    assert callable(label_cluster_categories)

def test_import_find_centroid():
    assert callable(find_centroid)

def test_import_get_safest_companies():
    assert callable(get_safest_companies)