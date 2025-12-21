import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FraudFeatureEngineer:
    """
    Feature engineering pipeline for Fraud_Data.csv.

    Provides methods to:
    - Parse timestamp columns
    - Add time-based features (hour of day, day of week)
    - Calculate time since signup
    - Compute transaction velocity features (total transactions per user, rolling 24h count)
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the FraudFeatureEngineer.

        Args:
            df (pd.DataFrame): Input fraud transaction dataset.

        Raises:
            TypeError: If df is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df)}")
        self.df: pd.DataFrame = df.copy()
        logger.info("Initialized FraudFeatureEngineer with %d rows", len(self.df))

    def parse_timestamps(self) -> "FraudFeatureEngineer":
        """
        Convert 'signup_time' and 'purchase_time' columns to datetime.

        Returns:
            FraudFeatureEngineer: self for method chaining.

        Raises:
            KeyError: If required columns are missing.
            ValueError: If datetime conversion fails.
        """
        try:
            logger.info("Parsing signup_time and purchase_time to datetime")
            self.df['signup_time'] = pd.to_datetime(self.df['signup_time'], errors='raise')
            self.df['purchase_time'] = pd.to_datetime(self.df['purchase_time'], errors='raise')
        except KeyError as e:
            raise KeyError(f"Missing required column: {e}")
        except Exception as e:
            raise ValueError(f"Failed to parse timestamps: {e}")
        return self

    def add_time_features(self) -> "FraudFeatureEngineer":
        """
        Add time-based features:
        - hour_of_day: hour of the purchase
        - day_of_week: day of the week (0=Monday, 6=Sunday)

        Returns:
            FraudFeatureEngineer: self for method chaining.

        Raises:
            KeyError: If 'purchase_time' column is missing.
        """
        if 'purchase_time' not in self.df.columns:
            raise KeyError("'purchase_time' column is required for time features")
        logger.info("Adding hour_of_day and day_of_week features")
        self.df['hour_of_day'] = self.df['purchase_time'].dt.hour
        self.df['day_of_week'] = self.df['purchase_time'].dt.dayofweek
        return self

    def add_time_since_signup(self) -> "FraudFeatureEngineer":
        """
        Add 'time_since_signup' feature in hours.

        Returns:
            FraudFeatureEngineer: self for method chaining.

        Raises:
            KeyError: If 'signup_time' or 'purchase_time' is missing.
        """
        if 'signup_time' not in self.df.columns or 'purchase_time' not in self.df.columns:
            raise KeyError("'signup_time' and 'purchase_time' columns are required for time_since_signup")
        logger.info("Adding time_since_signup feature (hours)")
        self.df['time_since_signup'] = (
            self.df['purchase_time'] - self.df['signup_time']
        ).dt.total_seconds() / 3600
        return self

    def add_transaction_velocity(self) -> "FraudFeatureEngineer":
        """
        Add transaction velocity features:
        - transactions_per_user: total transactions per user
        - transactions_last_24h: rolling 24-hour transaction count per user

        Returns:
            FraudFeatureEngineer: self for method chaining.

        Raises:
            KeyError: If 'user_id' or 'purchase_time' columns are missing.
        """
        if 'user_id' not in self.df.columns or 'purchase_time' not in self.df.columns:
            raise KeyError("'user_id' and 'purchase_time' columns are required for transaction velocity features")

        logger.info("Adding transaction velocity features")

        # Sort for rolling calculations
        self.df = self.df.sort_values(['user_id', 'purchase_time'])

        # Total transactions per user
        self.df['transactions_per_user'] = self.df.groupby('user_id')['user_id'].transform('count')

        # Helper column for rolling count
        self.df['_txn'] = 1

        # Set datetime index
        self.df = self.df.set_index('purchase_time')

        # Rolling 24-hour transaction count per user
        self.df['transactions_last_24h'] = (
            self.df.groupby('user_id')['_txn']
            .rolling('24H')
            .count()
            .reset_index(level=0, drop=True)
        )

        # Cleanup
        self.df = self.df.reset_index().drop(columns=['_txn'])
        logger.info("Transaction velocity features added successfully")
        return self

    def get_features(self) -> pd.DataFrame:
        """
        Return the DataFrame with all engineered features.

        Returns:
            pd.DataFrame: Feature-engineered dataset.
        """
        logger.info("Returning engineered feature dataframe")
        return self.df
