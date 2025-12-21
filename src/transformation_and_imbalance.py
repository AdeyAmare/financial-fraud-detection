import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FraudDataTransformer:
    """
    Transform numerical and categorical features for Fraud_Data.csv
    Includes optional SMOTE for handling class imbalance.
    """

    def __init__(self, df: pd.DataFrame, target: str, numeric_features: list, categorical_features: list, test_size=0.2, random_state=42):
        self.df = df.copy()
        self.target = target
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.test_size = test_size
        self.random_state = random_state

        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        logger.info("Initialized FraudDataTransformer with %d rows", len(self.df))

    def split_data(self):
        logger.info("Splitting data into train/test sets")
        X = self.df[self.numeric_features + self.categorical_features]
        y = self.df[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        logger.info("Train size: %d, Test size: %d", len(self.X_train), len(self.X_test))
        return self

    def transform_features(self):
        logger.info("Transforming features: scaling numeric, encoding categorical")

        transformers = []

        if self.numeric_features:
            transformers.append(('num', StandardScaler(), self.numeric_features))
        if self.categorical_features:
            # Use sparse_output=True to save memory
            transformers.append(('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), self.categorical_features))

        self.preprocessor = ColumnTransformer(transformers)

        logger.info("Fitting and transforming training data")
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)

        logger.info("Features transformed successfully")
        return self

    def handle_imbalance(self, strategy='SMOTE'):
        if strategy == 'SMOTE':
            logger.info("Applying SMOTE to handle class imbalance on training data")
            smote = SMOTE(random_state=self.random_state)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            logger.info("After SMOTE: Class distribution:\n%s", pd.Series(self.y_train).value_counts())
        else:
            logger.warning("No resampling applied; strategy=%s", strategy)
        return self

   
    def get_train_test(self):
        return self.X_train, self.X_test, self.y_train, self.y_test