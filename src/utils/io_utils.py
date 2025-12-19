import pandas as pd
from pathlib import Path

def save_dataframe(df: pd.DataFrame, path: str, index: bool = False):
    """
    Save a DataFrame to CSV safely.
    - Creates parent folders if they don't exist
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=index)
    print(f"[IO] Data saved to {file_path}")
