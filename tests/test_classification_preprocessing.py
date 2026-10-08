import pandas as pd
import numpy as np
from stock_chatbot.classification_preprocessing import preprocess_classification_features

def test_preprocess_classification_features():
    dates = pd.date_range(start='2020-01-01', periods=150, freq='D')
    df = pd.DataFrame({
        'close_value': np.linspace(100, 200, 150) + np.random.randn(150),
        'volume': np.random.randint(1000, 5000, size=150)
    }, index=dates)

    result = preprocess_classification_features(df)

    assert isinstance(result, pd.DataFrame)
    assert 'ma_126' in result.columns
    assert not result['ma_126'].isnull().all()
