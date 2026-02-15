import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE
from scipy.stats import randint
import matplotlib.pyplot as plt

# Optional ensemble support
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# -----------------------------
# Constants
# -----------------------------
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RF_N_ESTIMATORS = 100
DEFAULT_RF_MAX_DEPTH = 10
DEFAULT_CV_SPLITS = 5
DEFAULT_LR_MAX_ITER = 1000
DEFAULT_SMOTE = True

# -----------------------------
# Custom Exceptions
# -----------------------------
class PipelineError(Exception):
    """Custom exception for pipeline errors."""
    pass

# -----------------------------
# Utility Functions
# -----------------------------
def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Validate DataFrame schema."""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise PipelineError(f"Missing required columns: {missing}")

def plot_precision_recall_curve(y_true: pd.Series, y_probs: np.ndarray, model_name: str) -> None:
    """Plot precision-recall curve for evaluation."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, marker='.', label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve: {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

def apply_smote(X: np.ndarray, y: pd.Series, random_state: int = DEFAULT_RANDOM_STATE) -> Tuple[np.ndarray, pd.Series]:
    """Apply SMOTE to handle class imbalance."""
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

# -----------------------------
# Dataclass Config
# -----------------------------
@dataclass
class PipelineConfig:
    numeric_features: List[str]
    categorical_features: Optional[List[str]] = None
    target_col: str = 'class'
    use_smote: bool = DEFAULT_SMOTE
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = DEFAULT_RANDOM_STATE

# -----------------------------
# Production-Ready ModelingPipeline
# -----------------------------
class ModelingPipeline:
    """
    Production-ready ML pipeline with optional SMOTE and ensemble support.

    Supports Logistic Regression, Random Forest, XGBoost, LightGBM.
    Handles preprocessing of numeric and categorical features using standard scaling and one-hot encoding.
    Supports enriched evaluation artifacts: precision-recall curves and threshold analysis.
    """

    def __init__(self, df: pd.DataFrame, config: PipelineConfig):
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

        try:
            validate_dataframe(df, [config.target_col] + config.numeric_features + (config.categorical_features or []))
        except Exception as e:
            self.logger.error(f"Schema validation failed: {e}")
            raise PipelineError(e)

        self.df = df.copy()
        self.config = config
        self.numeric_features = config.numeric_features
        self.categorical_features = config.categorical_features or []
        self.target_col = config.target_col
        self.use_smote = config.use_smote
        self.test_size = config.test_size
        self.random_state = config.random_state

        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.results: List[Dict[str, Any]] = []

        transformers = [("num", StandardScaler(), self.numeric_features)]
        if self.categorical_features:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_features))
        self.preprocessor = ColumnTransformer(transformers=transformers)

        self.logger.info(f"Initialized pipeline with {len(df)} rows")

    # -----------------------------
    # Data Preparation
    # -----------------------------
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        try:
            self.logger.info("Step 1: Preparing data...")
            X = self.df[self.numeric_features + self.categorical_features]
            y = self.df[self.target_col]

            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, stratify=y, test_size=self.test_size, random_state=self.random_state
            )

            self.X_train = self.preprocessor.fit_transform(X_train_raw)
            self.X_test = self.preprocessor.transform(X_test_raw)
            self.y_train = y_train.reset_index(drop=True)
            self.y_test = y_test.reset_index(drop=True)

            self.logger.info(f"Training shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
            return self.X_train, self.X_test, self.y_train, self.y_test
        except Exception as e:
            self.logger.error(f"Error in prepare_data: {e}")
            raise PipelineError(e)

    # -----------------------------
    # Evaluate Model
    # -----------------------------
    def evaluate(self, model: Any) -> Dict[str, Any]:
        try:
            y_probs = model.predict_proba(self.X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(self.X_test)
            y_preds = model.predict(self.X_test)

            metrics = {
                "AUC_PR": average_precision_score(self.y_test, y_probs),
                "F1": f1_score(self.y_test, y_preds),
                "Precision": precision_score(self.y_test, y_preds),
                "Recall": recall_score(self.y_test, y_preds),
                "ConfusionMatrix": confusion_matrix(self.y_test, y_preds)
            }

            plot_precision_recall_curve(self.y_test, y_probs, model_name=type(model).__name__)
            return metrics
        except Exception as e:
            self.logger.error(f"Error in evaluate: {e}")
            raise PipelineError(e)

    # -----------------------------
    # Cross Validation
    # -----------------------------
    def cross_validate(self, model: Any, n_splits: int = DEFAULT_CV_SPLITS) -> Dict[str, float]:
        try:
            self.logger.info(f"Running Stratified {n_splits}-Fold CV...")
            X = self.df[self.numeric_features + self.categorical_features]
            y = self.df[self.target_col]
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

            f1_scores, auc_pr_scores = [], []

            for train_idx, val_idx in skf.split(X, y):
                X_train_fold_raw, X_val_fold_raw = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                X_train_fold = self.preprocessor.transform(X_train_fold_raw)
                X_val_fold = self.preprocessor.transform(X_val_fold_raw)

                if self.use_smote:
                    X_train_fold, y_train_fold = apply_smote(X_train_fold, y_train_fold, self.random_state)

                model.fit(X_train_fold, y_train_fold)
                y_probs = model.predict_proba(X_val_fold)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_val_fold)
                y_preds = model.predict(X_val_fold)

                f1_scores.append(f1_score(y_val_fold, y_preds))
                auc_pr_scores.append(average_precision_score(y_val_fold, y_probs))

            return {
                "F1_mean": float(np.mean(f1_scores)),
                "F1_std": float(np.std(f1_scores)),
                "AUC_PR_mean": float(np.mean(auc_pr_scores)),
                "AUC_PR_std": float(np.std(auc_pr_scores)),
            }
        except Exception as e:
            self.logger.error(f"Error in cross_validate: {e}")
            raise PipelineError(e)

    # -----------------------------
    # Existing Model Methods
    # -----------------------------
    def train_random_forest(self) -> RandomForestClassifier:
        try:
            model = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1, max_depth=10
            )
            X_res, y_res = self.X_train, self.y_train
            if self.use_smote:
                X_res, y_res = apply_smote(X_res, y_res, self.random_state)
            model.fit(X_res, y_res)
            metrics = self.evaluate(model)
            cv_metrics = self.cross_validate(model)
            self.results.append({"Model": "Random Forest", **metrics, **cv_metrics})
            return model
        except Exception as e:
            self.logger.error(f"Error in train_random_forest: {e}")
            raise PipelineError(e)

    def tune_and_train_logistic_regression(self, param_grid: Optional[Dict[str, Any]] = None, n_splits: int = 5) -> LogisticRegression:
        try:
            if param_grid is None:
                param_grid = {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs", "liblinear"]}

            lr = LogisticRegression(max_iter=1000, random_state=self.random_state, class_weight='balanced')
            grid = GridSearchCV(
                lr, param_grid, scoring="average_precision",
                cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state),
                n_jobs=-1
            )

            X_res, y_res = self.X_train, self.y_train
            if self.use_smote:
                X_res, y_res = apply_smote(X_res, y_res, self.random_state)

            grid.fit(X_res, y_res)
            best_lr = grid.best_estimator_
            metrics = self.evaluate(best_lr)
            cv_metrics = self.cross_validate(best_lr, n_splits=n_splits)
            self.results.append({"Model": "Logistic Regression (Tuned)", **metrics, **cv_metrics})
            logging.info(f"Logistic Regression best params: {grid.best_params_}")
            return best_lr
        except Exception as e:
            logging.error(f"Error in tune_logistic_regression: {e}")
            raise PipelineError(e)

    def tune_and_train_random_forest(self, n_iter: int = 10, n_splits: int = 3) -> RandomForestClassifier:
        try:
            param_dist = {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": randint(2, 11),
                "min_samples_leaf": randint(1, 5),
                "max_features": ["sqrt", "log2"]
            }

            rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            random_search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_dist,
                n_iter=n_iter, 
                scoring="average_precision",
                cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state),
                n_jobs=-1,
                random_state=self.random_state
            )

            X_res, y_res = self.X_train, self.y_train
            if self.use_smote:
                X_res, y_res = apply_smote(X_res, y_res, self.random_state)

            logging.info(f"Starting RandomizedSearch with {n_iter} iterations...")
            random_search.fit(X_res, y_res)
            best_rf = random_search.best_estimator_
            metrics = self.evaluate(best_rf)
            cv_metrics = self.cross_validate(best_rf, n_splits=n_splits)
            self.results.append({"Model": "Random Forest (Tuned)", **metrics, **cv_metrics})
            logging.info(f"Random Forest best params: {random_search.best_params_}")
            return best_rf
        except Exception as e:
            logging.error(f"Error in tune_random_forest: {e}")
            raise PipelineError(e)

    # -----------------------------
    # New Ensemble Methods
    # -----------------------------
    def train_xgboost(self, params: Optional[Dict[str, Any]] = None) -> Any:
        if xgb is None:
            raise PipelineError("XGBoost is not installed.")
        try:
            params = params or {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "random_state": self.random_state}
            model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
            X_res, y_res = self.X_train, self.y_train
            if self.use_smote:
                X_res, y_res = apply_smote(X_res, y_res, self.random_state)
            model.fit(X_res, y_res)
            metrics = self.evaluate(model)
            cv_metrics = self.cross_validate(model)
            self.results.append({"Model": "XGBoost", **metrics, **cv_metrics})
            return model
        except Exception as e:
            self.logger.error(f"Error in train_xgboost: {e}")
            raise PipelineError(e)

    def train_lightgbm(self, params: Optional[Dict[str, Any]] = None) -> Any:
        if lgb is None:
            raise PipelineError("LightGBM is not installed.")
        try:
            params = params or {"n_estimators": 100, "max_depth": -1, "learning_rate": 0.1, "random_state": self.random_state}
            model = lgb.LGBMClassifier(**params)
            X_res, y_res = self.X_train, self.y_train
            if self.use_smote:
                X_res, y_res = apply_smote(X_res, y_res, self.random_state)
            model.fit(X_res, y_res)
            metrics = self.evaluate(model)
            cv_metrics = self.cross_validate(model)
            self.results.append({"Model": "LightGBM", **metrics, **cv_metrics})
            return model
        except Exception as e:
            self.logger.error(f"Error in train_lightgbm: {e}")
            raise PipelineError(e)

    # -----------------------------
    # Compare & Select
    # -----------------------------
    def compare_models(self) -> pd.DataFrame:
        try:
            self.logger.info("Comparing models...")
            df_results = pd.DataFrame(self.results)
            return df_results.sort_values(by="AUC_PR", ascending=False).reset_index(drop=True)
        except Exception as e:
            self.logger.error(f"Error in compare_models: {e}")
            raise PipelineError(e)

    def select_best_model(self) -> Tuple[pd.Series, str]:
        try:
            comparison = self.compare_models()
            best = comparison.iloc[0]
            justification = (
                f"{best['Model']} selected due to highest AUC-PR "
                f"({best['AUC_PR']:.3f}) and strong recall, prioritizing "
                f"undetected fraud reduction while maintaining interpretability."
            )
            self.logger.info(justification)
            return best, justification
        except Exception as e:
            self.logger.error(f"Error in select_best_model: {e}")
            raise PipelineError(e)
