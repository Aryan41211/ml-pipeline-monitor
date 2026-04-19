import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 100.0],
            "f2": [0.0, 0.0, 0.0, 0.0],
            "f3": [5.0, np.nan, 7.0, 8.0],
        }
    )


@pytest.fixture
def sample_target_series() -> pd.Series:
    return pd.Series([0, 0, 0, 1], name="target")
