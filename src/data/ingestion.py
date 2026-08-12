"""
Data Ingestion Pipeline for ChurnGuard AI.
Loads raw dataset and logs schema details.
"""
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
from .download import download_raw_dataset
from ..utils.config import load_config, get_project_root
from ..utils.logger import get_logger

logger = get_logger("DataIngestion")


def ingest_raw_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads raw customer dataset and returns DataFrame along with ingestion summary metadata.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Raw DataFrame and metadata dictionary.
    """
    raw_path, df = download_raw_dataset()

    metadata = {
        "raw_path": str(raw_path),
        "file_size_bytes": raw_path.stat().st_size,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }

    logger.info(f"Data ingestion complete. Metadata: {metadata['n_rows']} rows x {metadata['n_cols']} cols")
    return df, metadata


if __name__ == "__main__":
    ingest_raw_data()
