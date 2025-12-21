import pandas as pd
from pathlib import Path
from typing import Any


class FraudDataLoader:
    """
    Loader for fraud transaction data CSV files.

    Ensures the presence of required columns and parses datetime columns.

    Required Columns:
        - signup_time
        - purchase_time
    """

    REQUIRED_COLUMNS = {"signup_time", "purchase_time"}

    def __init__(self, file_path: str):
        """
        Initialize the FraudDataLoader with a CSV file path.

        Args:
            file_path (str): Path to the CSV file.
        """
        self.file_path: Path = Path(file_path)

    def load(self) -> pd.DataFrame:
        """
        Load the fraud dataset as a pandas DataFrame.

        Performs:
        - File existence check
        - CSV reading with pandas
        - Validation of required columns
        - Conversion of datetime columns

        Returns:
            pd.DataFrame: Loaded and validated dataset.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            RuntimeError: If reading the CSV file fails.
            ValueError: If required columns are missing or datetime parsing fails.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        try:
            df: pd.DataFrame = pd.read_csv(self.file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV file: {e}")

        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        try:
            df["signup_time"] = pd.to_datetime(df["signup_time"], errors="raise")
            df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="raise")
        except Exception as e:
            raise ValueError(f"Datetime parsing failed: {e}")

        return df


class CreditCardDataLoader:
    """
    Loader for credit card data CSV files.
    """

    def __init__(self, file_path: str):
        """
        Initialize the CreditCardDataLoader with a CSV file path.

        Args:
            file_path (str): Path to the CSV file.
        """
        self.file_path: Path = Path(file_path)

    def load(self) -> pd.DataFrame:
        """
        Load the credit card dataset as a pandas DataFrame.

        Performs:
        - File existence check
        - CSV reading with pandas

        Returns:
            pd.DataFrame: Loaded dataset.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            RuntimeError: If reading the CSV file fails.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        try:
            return pd.read_csv(self.file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV file: {e}")


class IPCountryLoader:
    """
    Loader for IP-to-country mapping CSV files.

    Ensures the presence of required columns:
        - lower_bound_ip_address
        - upper_bound_ip_address
        - country
    """

    REQUIRED_COLUMNS = {
        "lower_bound_ip_address",
        "upper_bound_ip_address",
        "country",
    }

    def __init__(self, file_path: str):
        """
        Initialize the IPCountryLoader with a CSV file path.

        Args:
            file_path (str): Path to the CSV file.
        """
        self.file_path: Path = Path(file_path)

    def load(self) -> pd.DataFrame:
        """
        Load the IP-to-country dataset as a pandas DataFrame.

        Performs:
        - File existence check
        - CSV reading with pandas
        - Validation of required columns

        Returns:
            pd.DataFrame: Loaded and validated dataset.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            RuntimeError: If reading the CSV file fails.
            ValueError: If required columns are missing.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        try:
            df: pd.DataFrame = pd.read_csv(self.file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV file: {e}")

        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df
