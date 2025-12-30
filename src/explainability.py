import joblib
import logging
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import Optional

# Ensure JavaScript is initialized for SHAP plots in Notebooks
shap.initjs()

# ----------------------------
# Logging configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ModelExplainability:
    """
    ModelExplainability provides SHAP-based interpretability utilities
    for trained fraud detection models that rely on a saved preprocessing
    pipeline.

    This class supports:
    - Built-in feature importance visualization
    - Global SHAP summary plots
    - Instance-level SHAP force plots (TP / FP / FN)
    - Extraction of top SHAP feature drivers

    Notes
    -----
    - The model is assumed to be already trained.
    - The preprocessor must expose `transform()` and `get_feature_names_out()`.
    - SHAP logic is intentionally defensive to handle tree and non-tree models.
    """

    def __init__(
        self,
        model_path: str,
        preprocessor_path: str,
        X: pd.DataFrame,
        y: pd.Series
    ):
        """
        Initialize the explainability module.

        Parameters
        ----------
        model_path : str
            Path to the saved trained model (joblib).
        preprocessor_path : str
            Path to the saved preprocessing pipeline (joblib).
        X : pd.DataFrame
            Raw feature dataframe (before preprocessing).
        y : pd.Series
            Ground-truth labels aligned with X.

        Raises
        ------
        TypeError
            If X or y are not pandas objects.
        RuntimeError
            If model or preprocessor loading fails.
        """
        logging.info("Initializing ModelExplainability")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            raise TypeError("y must be a pandas Series or DataFrame")

        try:
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
        except Exception as e:
            logging.error("Failed to load model or preprocessor", exc_info=True)
            raise RuntimeError("Model or preprocessor loading failed") from e

        self.X_raw = X.copy()
        self.y = y.reset_index(drop=True)

        try:
            self.X = self.preprocessor.transform(X)
            self.feature_names = self.preprocessor.get_feature_names_out()
        except Exception as e:
            logging.error("Preprocessing failed", exc_info=True)
            raise RuntimeError("Error during feature preprocessing") from e

        try:
            self.predictions = self.model.predict(self.X)
        except Exception as e:
            logging.error("Prediction failed", exc_info=True)
            raise RuntimeError("Model prediction failed") from e

        self.probabilities = (
            self.model.predict_proba(self.X)[:, 1]
            if hasattr(self.model, "predict_proba")
            else None
        )

        self.explainer: Optional[object] = None
        self.shap_values: Optional[np.ndarray] = None
        self.X_shap: Optional[np.ndarray] = None

        logging.info("ModelExplainability initialized successfully")

    def plot_builtin_feature_importance(self, top_n: int = 10):
        """
        Plot built-in feature importances for tree-based models.

        Parameters
        ----------
        top_n : int, default=10
            Number of top features to display.

        Returns
        -------
        pd.DataFrame or None
            DataFrame of top features if supported, otherwise None.
        """
        if not hasattr(self.model, "feature_importances_"):
            logging.warning("Model does not support built-in feature importance")
            return None

        importances = self.model.feature_importances_

        if len(importances) != len(self.feature_names):
            logging.warning("Feature importance and feature name length mismatch")

        importance_df = (
            pd.DataFrame({
                "feature": self.feature_names,
                "importance": importances
            })
            .sort_values(by="importance", ascending=False)
            .head(top_n)
        )

        plt.figure(figsize=(8, 6))
        plt.barh(
            importance_df["feature"][::-1],
            importance_df["importance"][::-1]
        )
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Feature Importances (Built-in)")
        plt.tight_layout()
        plt.show()

        return importance_df

    def compute_shap_values(self, sample_size: int = 500):
        """
        Compute SHAP values for the model.

        Parameters
        ----------
        sample_size : int, default=500
            Maximum number of samples used for SHAP computation.
        """
        logging.info("Computing SHAP values")

        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError("sample_size must be a positive integer")

        try:
            if self.X.shape[0] > sample_size:
                self.X_shap = shap.sample(
                    self.X, sample_size, random_state=42
                )
            else:
                self.X_shap = self.X

            if hasattr(self.model, "estimators_"):
                logging.info("Using TreeExplainer")
                self.explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.explainer.shap_values(self.X_shap)
                if isinstance(self.shap_values, list):
                    self.shap_values = self.shap_values[1]
            else:
                logging.info("Using model-agnostic SHAP Explainer")
                background = shap.sample(self.X, 100, random_state=42)
                self.explainer = shap.Explainer(self.model, background)
                self.shap_values = self.explainer(self.X_shap)

        except Exception as e:
            logging.error("SHAP computation failed", exc_info=True)
            raise RuntimeError("Error during SHAP computation") from e

    def plot_shap_summary(self, max_display: int = 20):
        """
        Plot global SHAP summary plot.

        Parameters
        ----------
        max_display : int, default=20
            Maximum number of features to display.
        """
        if self.shap_values is None:
            self.compute_shap_values()

        shap_vals_to_plot = (
            self.shap_values.values
            if hasattr(self.shap_values, "values")
            else self.shap_values
        )

        shap.summary_plot(
            shap_vals_to_plot,
            features=self.X_shap,
            feature_names=self.feature_names,
            max_display=max_display
        )

    def plot_force_plot_for_case(self, case_type: str = "TP"):
        """
        Generate a SHAP force plot for a specific confusion-matrix case.

        Parameters
        ----------
        case_type : {"TP", "FP", "FN"}
            Type of prediction case.

        Returns
        -------
        shap.plots._force.AdditiveForceVisualizer or None
        """
        logging.info(f"Generating force plot for case: {case_type}")

        if case_type not in {"TP", "FP", "FN"}:
            raise ValueError("case_type must be one of ['TP', 'FP', 'FN']")

        y_true, y_pred = self.y.values, self.predictions

        if case_type == "TP":
            idx_list = np.where((y_true == 1) & (y_pred == 1))[0]
        elif case_type == "FP":
            idx_list = np.where((y_true == 0) & (y_pred == 1))[0]
        else:
            idx_list = np.where((y_true == 1) & (y_pred == 0))[0]

        if len(idx_list) == 0:
            logging.warning(f"No samples found for {case_type}")
            return None

        idx = idx_list[0]

        try:
            X_instance = self.X[idx]
            if hasattr(X_instance, "toarray"):
                X_instance = X_instance.toarray().flatten()
            else:
                X_instance = np.array(X_instance).flatten()
        except Exception as e:
            logging.error("Failed to extract instance features", exc_info=True)
            raise RuntimeError("Feature extraction for force plot failed") from e

        try:
            if hasattr(self.model, "estimators_") or "XGB" in str(type(self.model)):
                explainer = shap.TreeExplainer(self.model)
                raw_vals = explainer.shap_values(self.X[idx:idx + 1])

                if isinstance(raw_vals, list):
                    shap_vals = raw_vals[1].flatten()
                    base_val = explainer.expected_value[1]
                elif len(raw_vals.shape) == 3:
                    shap_vals = raw_vals[0, :, 1].flatten()
                    base_val = explainer.expected_value[1]
                else:
                    shap_vals = raw_vals.flatten()
                    base_val = explainer.expected_value
            else:
                background = shap.sample(self.X, 100, random_state=42)
                explainer = shap.Explainer(self.model, background)
                shap_exp = explainer(self.X[idx:idx + 1])

                if len(shap_exp.values.shape) == 3:
                    shap_vals = shap_exp.values[0, :, 1].flatten()
                    base_val = shap_exp.base_values[0, 1]
                else:
                    shap_vals = shap_exp.values[0].flatten()
                    base_val = shap_exp.base_values[0]

        except Exception as e:
            logging.error("SHAP force plot computation failed", exc_info=True)
            raise RuntimeError("Force plot SHAP computation failed") from e

        if len(shap_vals) == 2 * len(X_instance):
            logging.warning("Detected interleaved SHAP values, correcting dimensions")
            shap_vals = shap_vals[len(X_instance):]

        return shap.force_plot(
            base_value=float(base_val),
            shap_values=shap_vals,
            features=X_instance,
            feature_names=self.feature_names
        )

    def get_top_drivers(self, top_n: int = 5):
        """
        Retrieve top SHAP feature drivers by mean absolute contribution.

        Parameters
        ----------
        top_n : int, default=5
            Number of top drivers to return.

        Returns
        -------
        pd.DataFrame
            Top features ranked by mean absolute SHAP value.
        """
        if self.shap_values is None:
            self.compute_shap_values()

        shap_array = (
            self.shap_values.values
            if hasattr(self.shap_values, "values")
            else self.shap_values
        )

        if len(shap_array.shape) == 3:
            logging.info("Detected 3D SHAP array, selecting positive class (index 1)")
            shap_array = shap_array[:, :, 1]

        mean_abs_shap = np.abs(shap_array).mean(axis=0)
        mean_abs_shap = np.ravel(mean_abs_shap)

        return (
            pd.DataFrame({
                "feature": self.feature_names,
                "mean_abs_shap": mean_abs_shap
            })
            .sort_values(by="mean_abs_shap", ascending=False)
            .head(top_n)
        )
