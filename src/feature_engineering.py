import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FraudFeatureEngineer:
    """
    Feature engineering for Fraud_Data.csv
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        logger.info("Initialized FraudFeatureEngineer with %d rows", len(self.df))

    def parse_timestamps(self):
        logger.info("Parsing signup_time and purchase_time to datetime")
        self.df['signup_time'] = pd.to_datetime(self.df['signup_time'])
        self.df['purchase_time'] = pd.to_datetime(self.df['purchase_time'])
        return self

    def add_time_features(self):
        logger.info("Adding hour_of_day and day_of_week features")
        self.df['hour_of_day'] = self.df['purchase_time'].dt.hour
        self.df['day_of_week'] = self.df['purchase_time'].dt.dayofweek
        return self

    def add_time_since_signup(self):
        logger.info("Adding time_since_signup feature (hours)")
        self.df['time_since_signup'] = (
            self.df['purchase_time'] - self.df['signup_time']
        ).dt.total_seconds() / 3600
        return self

    def add_transaction_velocity(self):
        logger.info("Adding transaction velocity features")

        # Sort is required for rolling windows
        self.df = self.df.sort_values(['user_id', 'purchase_time'])

        # Total transactions per user
        self.df['transactions_per_user'] = (
            self.df.groupby('user_id')['user_id']
            .transform('count')
        )

        # Add helper column for rolling count
        self.df['_txn'] = 1

        # Set datetime index
        self.df = self.df.set_index('purchase_time')

        # Rolling 24-hour transaction count per user
        self.df['transactions_last_24h'] = (
            self.df
            .groupby('user_id')['_txn']
            .rolling('24H')
            .count()
            .reset_index(level=0, drop=True)
        )

        # Clean up
        self.df = self.df.reset_index()
        self.df = self.df.drop(columns=['_txn'])

        logger.info("Transaction velocity features added successfully")
        return self

    def get_features(self):
        logger.info("Returning engineered feature dataframe")
        return self.df
