import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from stock_chatbot.clustering_model import perform_clustering

def test_perform_clustering():
    # Create a small sample DataFrame with scaled features
    data = {
        'feature1': [0.1, 0.2, 0.3, 0.8, 0.9, 1.0],
        'feature2': [1.0, 0.9, 0.8, 0.3, 0.2, 0.1]
    }
    df = pd.DataFrame(data)

    model, labels = perform_clustering(df)

    # Check if the model is a KMeans instance
    assert isinstance(model, KMeans), "Returned model is not a KMeans instance"

    # Check if the number of labels equals number of samples
    assert len(labels) == len(df), "Number of labels does not match number of input rows"

    # Check that all labels are integers and in the correct range
    unique_labels = set(labels)
    assert unique_labels.issubset({0, 1, 2}), f"Unexpected cluster labels: {unique_labels}"
