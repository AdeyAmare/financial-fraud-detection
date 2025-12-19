import pandas as pd
from pathlib import Path


class FraudDataLoader:
    REQUIRED_COLUMNS = {"signup_time", "purchase_time"}

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        try:
            df = pd.read_csv(self.file_path)
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
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        try:
            return pd.read_csv(self.file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV file: {e}")


class IPCountryLoader:
    REQUIRED_COLUMNS = {
        "lower_bound_ip_address",
        "upper_bound_ip_address",
        "country",
    }

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        try:
            df = pd.read_csv(self.file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV file: {e}")

        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df
