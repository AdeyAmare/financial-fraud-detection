import pytest
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Tuple
from unittest.mock import MagicMock, patch

from src.explainability import ModelExplainability, ExplainabilityConfig

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# Constants
# -----------------------------
TOP_N_FEATURES: int = 2
SAMPLE_SIZE: int = 2

# -----------------------------
# Dataclasses
# -----------------------------
@dataclass
class MockData:
    X: pd.DataFrame
    y: pd.Series

@dataclass
class MockModel:
    model: MagicMock
    preprocessor: MagicMock

# -----------------------------
# Utility Functions
# -----------------------------
def create_mock_data() -> MockData:
    """
    Creates mock X, y dataset for explainability tests.
    """
    X = pd.DataFrame({
        "feature1": [0.1, 0.2, 0.3],
        "feature2": [1, 2, 3]
    })
    y = pd.Series([0, 1, 0])
    logger.info("Created mock data for explainability tests")
    return MockData(X=X, y=y)

def create_mock_model() -> MockModel:
    """
    Creates mock model and preprocessor for explainability tests.
    """
    model = MagicMock()
    model.predict.return_value = np.array([0, 1, 0])
    model.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    model.feature_importances_ = np.array([0.6, 0.4])

    preprocessor = MagicMock()
    preprocessor.transform.return_value = np.array([[0.1, 1], [0.2, 2], [0.3, 3]])
    preprocessor.get_feature_names_out.return_value = np.array(["feature1", "feature2"])

    logger.info("Created mock model and preprocessor for explainability tests")
    return MockModel(model=model, preprocessor=preprocessor)

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def mock_data() -> MockData:
    return create_mock_data()

@pytest.fixture
def mock_model() -> MockModel:
    return create_mock_model()

# -----------------------------
# Test Functions
# -----------------------------
@patch("joblib.load")
def test_model_explainability_init(
    mock_joblib_load, mock_data: MockData, mock_model: MockModel
) -> None:
    """
    Tests initialization of ModelExplainability and basic attribute setup.
    """
    # Patch joblib.load to return mocks
    mock_joblib_load.side_effect = [mock_model.model, mock_model.preprocessor]

    config = ExplainabilityConfig(top_n=TOP_N_FEATURES, sample_size=SAMPLE_SIZE)
    explainer = ModelExplainability(
        "dummy_model_path", "dummy_preprocessor_path", mock_data.X, mock_data.y, config=config
    )

    # Check basic attributes
    assert hasattr(explainer, "X_raw")
    assert hasattr(explainer, "predictions")
    assert hasattr(explainer, "feature_names")
    assert explainer.feature_names.tolist() == ["feature1", "feature2"]
    assert explainer.predictions.tolist() == [0, 1, 0]

    logger.info("ModelExplainability initialization test passed")

@patch("joblib.load")
def test_get_top_drivers_returns_dataframe(
    mock_joblib_load, mock_data: MockData, mock_model: MockModel
) -> None:
    """
    Tests that get_top_drivers() returns a DataFrame with top features by mean absolute SHAP values.
    """
    # Patch joblib.load to return mocks
    mock_joblib_load.side_effect = [mock_model.model, mock_model.preprocessor]

    config = ExplainabilityConfig()
    explainer = ModelExplainability(
        "dummy_model_path", "dummy_preprocessor_path", mock_data.X, mock_data.y, config=config
    )

    # Mock SHAP values directly (3 samples x 2 features)
    explainer.shap_values = np.array([[0.1, -0.2], [0.05, 0.1], [-0.1, 0.05]])

    top_drivers = explainer.get_top_drivers(top_n=TOP_N_FEATURES)

    assert isinstance(top_drivers, pd.DataFrame)
    assert top_drivers.shape[0] == TOP_N_FEATURES
    assert "feature" in top_drivers.columns
    assert "mean_abs_shap" in top_drivers.columns

    logger.info("get_top_drivers returned correct DataFrame structure")
