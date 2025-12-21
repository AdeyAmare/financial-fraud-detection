import pandas as pd
import logging
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FraudDataTransformer:
    """
    Transform numerical and categorical features for Fraud_Data.csv.

    Features:
    - Standard scaling for numeric columns
    - One-hot encoding for categorical columns
    - Train/test split
    - Optional SMOTE for class imbalance handling
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        numeric_features: List[str],
        categorical_features: List[str],
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize the FraudDataTransformer.

        Args:
            df (pd.DataFrame): Input dataset.
            target (str): Name of the target column.
            numeric_features (List[str]): Names of numeric feature columns.
            categorical_features (List[str]): Names of categorical feature columns.
            test_size (float, optional): Proportion of data for testing. Defaults to 0.2.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.

        Raises:
            TypeError: If df is not a pandas DataFrame.
            ValueError: If target or features are missing in the DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df)}")
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")
        missing_numeric = set(numeric_features) - set(df.columns)
        if missing_numeric:
            raise ValueError(f"Numeric features missing in DataFrame: {missing_numeric}")
        missing_categorical = set(categorical_features) - set(df.columns)
        if missing_categorical:
            raise ValueError(f"Categorical features missing in DataFrame: {missing_categorical}")

        self.df: pd.DataFrame = df.copy()
        self.target: str = target
        self.numeric_features: List[str] = numeric_features
        self.categorical_features: List[str] = categorical_features
        self.test_size: float = test_size
        self.random_state: int = random_state

        self.preprocessor: Optional[ColumnTransformer] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        logger.info("Initialized FraudDataTransformer with %d rows", len(self.df))

    def split_data(self) -> "FraudDataTransformer":
        """
        Split the dataset into training and testing sets.

        Returns:
            FraudDataTransformer: self for method chaining.
        """
        logger.info("Splitting data into train/test sets")
        X = self.df[self.numeric_features + self.categorical_features]
        y = self.df[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        logger.info("Train size: %d, Test size: %d", len(self.X_train), len(self.X_test))
        return self

    def transform_features(self) -> "FraudDataTransformer":
        """
        Apply transformations:
        - StandardScaler for numeric columns
        - OneHotEncoder for categorical columns

        Returns:
            FraudDataTransformer: self for method chaining.
        """
        logger.info("Transforming features: scaling numeric, encoding categorical")
        transformers = []

        if self.numeric_features:
            transformers.append(('num', StandardScaler(), self.numeric_features))
        if self.categorical_features:
            transformers.append(
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), self.categorical_features)
            )

        self.preprocessor = ColumnTransformer(transformers)

        logger.info("Fitting and transforming training data")
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)

        logger.info("Features transformed successfully")
        return self

    def handle_imbalance(self, strategy: str = 'SMOTE') -> "FraudDataTransformer":
        """
        Handle class imbalance in the training set.

        Args:
            strategy (str): Resampling strategy. Currently only 'SMOTE' is supported.

        Returns:
            FraudDataTransformer: self for method chaining.

        Raises:
            ValueError: If unknown strategy is provided.
        """
        if strategy == 'SMOTE':
            logger.info("Applying SMOTE to handle class imbalance on training data")
            smote = SMOTE(random_state=self.random_state)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            logger.info("After SMOTE: Class distribution:\n%s", pd.Series(self.y_train).value_counts())
        elif strategy is not None:
            raise ValueError(f"Unknown imbalance strategy: {strategy}")
        else:
            logger.warning("No resampling applied; strategy=None")
        return self

    def get_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Retrieve the transformed train/test datasets.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
        """
        return self.X_train, self.X_test, self.y_train, self.y_test
