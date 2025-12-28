import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from scipy.stats import randint


class ModelingPipeline:
    """
    A machine learning pipeline for training, evaluating, tuning, and comparing classification models.
    
    Supports Logistic Regression and Random Forest with optional SMOTE oversampling. 
    Handles preprocessing of numeric and categorical features using standard scaling and one-hot encoding.
    """

    def __init__(self, 
             df: pd.DataFrame, 
             numeric_features: List[str], 
             categorical_features: Optional[List[str]] = None,
             target_col: str = 'class', 
             use_smote: bool = True, 
             test_size: float = 0.2, 
             random_state: int = 42) -> None:
        try:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("df must be a pandas DataFrame.")
            if not numeric_features or not isinstance(numeric_features, list):
                raise ValueError("numeric_features must be a non-empty list of column names.")
            if categorical_features is not None and not isinstance(categorical_features, list):
                raise ValueError("categorical_features must be a list of column names or None.")
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        except Exception as e:
            logging.error(f"Error in __init__: {e}")
            raise

        self.df = df
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features or []  # default to empty list
        self.target_col = target_col
        self.use_smote = use_smote
        self.test_size = test_size
        self.random_state = random_state

        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.results: List[Dict[str, Any]] = []

        # Build ColumnTransformer dynamically
        transformers = [("num", StandardScaler(), self.numeric_features)]
        if self.categorical_features:
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_features)
            )
        
        self.preprocessor = ColumnTransformer(transformers=transformers)

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """Splits the dataset into training and test sets and applies preprocessing."""
        try:
            logging.info("Step 1: Preparing data...")
            X = self.df[self.numeric_features + self.categorical_features]
            y = self.df[self.target_col]

            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, stratify=y, test_size=self.test_size, random_state=self.random_state
            )

            self.X_train = self.preprocessor.fit_transform(X_train_raw)
            self.X_test = self.preprocessor.transform(X_test_raw)
            self.y_train = y_train.reset_index(drop=True)
            self.y_test = y_test.reset_index(drop=True)

            logging.info(f"Training shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
            return self.X_train, self.X_test, self.y_train, self.y_test
        except Exception as e:
            logging.error(f"Error in prepare_data: {e}")
            raise

    def evaluate(self, model: Any) -> Dict[str, Any]:
        """Evaluates a trained model on the test set."""
        try:
            y_probs = model.predict_proba(self.X_test)[:, 1]
            y_preds = model.predict(self.X_test)

            metrics = {
                "AUC_PR": average_precision_score(self.y_test, y_probs),
                "F1": f1_score(self.y_test, y_preds),
                "Precision": precision_score(self.y_test, y_preds),
                "Recall": recall_score(self.y_test, y_preds),
                "ConfusionMatrix": confusion_matrix(self.y_test, y_preds)
            }
            return metrics
        except Exception as e:
            logging.error(f"Error in evaluate: {e}")
            raise

    def cross_validate(self, model: Any, n_splits: int = 5) -> Dict[str, float]:
        """Performs Stratified K-Fold CV and returns mean and std for F1 and AUC-PR."""
        try:
            logging.info(f"Running Stratified {n_splits}-Fold CV...")
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
                    smote = SMOTE(random_state=self.random_state)
                    X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)

                model.fit(X_train_fold, y_train_fold)
                y_probs = model.predict_proba(X_val_fold)[:, 1]
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
            logging.error(f"Error in cross_validate: {e}")
            raise

    def train_random_forest(self) -> RandomForestClassifier:
        """Trains a Random Forest model with optional SMOTE."""
        try:
            model = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1, max_depth=10
            )
            X_res, y_res = self.X_train, self.y_train
            if self.use_smote:
                smote = SMOTE(random_state=self.random_state)
                X_res, y_res = smote.fit_resample(X_res, y_res)
            model.fit(X_res, y_res)
            metrics = self.evaluate(model)
            cv_metrics = self.cross_validate(model)
            self.results.append({"Model": "Random Forest", **metrics, **cv_metrics})
            return model
        except Exception as e:
            logging.error(f"Error in train_random_forest: {e}")
            raise

    def tune_and_train_logistic_regression(self, param_grid: Optional[Dict[str, Any]] = None, n_splits: int = 5) -> LogisticRegression:
        """Hyperparameter tuning for Logistic Regression using GridSearchCV."""
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
                smote = SMOTE(random_state=self.random_state)
                X_res, y_res = smote.fit_resample(X_res, y_res)

            grid.fit(X_res, y_res)
            best_lr = grid.best_estimator_
            metrics = self.evaluate(best_lr)
            cv_metrics = self.cross_validate(best_lr, n_splits=n_splits)
            self.results.append({"Model": "Logistic Regression (Tuned)", **metrics, **cv_metrics})
            logging.info(f"Logistic Regression best params: {grid.best_params_}")
            return best_lr
        except Exception as e:
            logging.error(f"Error in tune_logistic_regression: {e}")
            raise

    def tune_and_train_random_forest(self, n_iter: int = 10, n_splits: int = 3) -> RandomForestClassifier:
        """
        Hyperparameter tuning for Random Forest using RandomizedSearchCV for speed.
        
        Args:
            n_iter: Number of parameter settings that are sampled. 
            n_splits: Number of folds for cross-validation (reduced for speed).
        """
        try:
            # Using distributions for more flexible sampling
            param_dist = {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": randint(2, 11),
                "min_samples_leaf": randint(1, 5),
                "max_features": ["sqrt", "log2"]
            }

            rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            
            # RandomizedSearchCV samples 'n_iter' combinations instead of all of them
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
                smote = SMOTE(random_state=self.random_state)
                X_res, y_res = smote.fit_resample(X_res, y_res)

            logging.info(f"Starting RandomizedSearch with {n_iter} iterations...")
            random_search.fit(X_res, y_res)
            
            best_rf = random_search.best_estimator_
            metrics = self.evaluate(best_rf)
            
            # We use the original CV method to get consistent final metrics for comparison
            cv_metrics = self.cross_validate(best_rf, n_splits=n_splits)
            
            self.results.append({"Model": "Random Forest (Tuned)", **metrics, **cv_metrics})
            logging.info(f"Random Forest best params: {random_search.best_params_}")
            
            return best_rf
        except Exception as e:
            logging.error(f"Error in tune_random_forest: {e}")
            raise
        
    def compare_models(self) -> pd.DataFrame:
        """Compares trained models based on evaluation metrics (AUC-PR)."""
        try:
            logging.info("Comparing models...")
            df_results = pd.DataFrame(self.results)
            return df_results.sort_values(by="AUC_PR", ascending=False).reset_index(drop=True)
        except Exception as e:
            logging.error(f"Error in compare_models: {e}")
            raise

    def select_best_model(self) -> Tuple[pd.Series, str]:
        """Selects the best model based on AUC-PR and recalls metric variability."""
        try:
            comparison = self.compare_models()
            best = comparison.iloc[0]
            justification = (
                f"{best['Model']} selected due to highest AUC-PR "
                f"({best['AUC_PR']:.3f}) and strong recall, prioritizing "
                f"undetected fraud reduction while maintaining interpretability."
            )
            logging.info(justification)
            return best, justification
        except Exception as e:
            logging.error(f"Error in select_best_model: {e}")
            raise
