import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class TransactionDataCleaner:
    """
    Clean transaction datasets:
    - Remove duplicates
    - Handle missing values
    - Correct data types
    - Report dataset issues (duplicates, missing values, dtypes)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.logger = logging.getLogger(self.__class__.__name__)

    # ---------------------------
    # 1️⃣ Report method
    # ---------------------------
    def report(self):
        """
        Log dataset info:
        - Shape
        - Data types
        - Duplicate rows
        - Missing values count and percentage
        """
        self.logger.info(f"Dataset shape: {self.df.shape}")
        self.logger.info(f"Column data types:\n{self.df.dtypes}")
        
        # Duplicates
        duplicate_count = self.df.duplicated().sum()
        self.logger.info(f"Number of duplicate rows: {duplicate_count}")
        
        # Missing values
        missing_per_column = self.df.isna().sum()
        total_missing = missing_per_column.sum()
        missing_pct = (missing_per_column / len(self.df)) * 100
        
        self.logger.info(f"Missing values per column:\n{missing_per_column}")
        self.logger.info(f"Missing values (%) per column:\n{missing_pct}")
        self.logger.info(f"Total missing values: {total_missing}")

    # ---------------------------
    # 2️⃣ Remove duplicates
    # ---------------------------
    def remove_duplicates(self) -> pd.DataFrame:
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial_rows - len(self.df)
        self.logger.info(f"Removed {removed} duplicate rows")
        return self.df

    # ---------------------------
    # 3️⃣ Handle missing values
    # ---------------------------
    def handle_missing_values(self, strategy: str = "drop") -> pd.DataFrame:
        missing_before = self.df.isna().sum().sum()
        if missing_before == 0:
            self.logger.info("No missing values detected")
            return self.df

        if strategy == "drop":
            self.df = self.df.dropna()
        elif strategy == "fill_zero":
            self.df = self.df.fillna(0)
        elif strategy == "fill_median":
            numeric_cols = self.df.select_dtypes(include="number").columns
            for col in numeric_cols:
                median = self.df[col].median()
                self.df[col].fillna(median, inplace=True)
        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")

        missing_after = self.df.isna().sum().sum()
        self.logger.info(f"Missing values before: {missing_before}, after: {missing_after}")
        return self.df

    # ---------------------------
    # 4️⃣ Correct data types
    # ---------------------------
    def correct_dtypes(self) -> pd.DataFrame:
        if "age" in self.df.columns:
            try:
                self.df["age"] = self.df["age"].astype(int)
            except Exception as e:
                self.logger.warning(f"Could not convert 'age' to int: {e}")
        return self.df

    # ---------------------------
    # 5️⃣ Clean pipeline
    # ---------------------------
    def clean(self, missing_strategy: str = "drop") -> pd.DataFrame:
        self.remove_duplicates()
        self.handle_missing_values(strategy=missing_strategy)
        self.correct_dtypes()
        return self.df
