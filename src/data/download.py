"""
Dataset Download Utility for ChurnGuard AI.
Downloads the official IBM Telco Customer Churn dataset from public source if missing.
"""
from pathlib import Path
import urllib.request
from typing import Tuple
import pandas as pd
from ..utils.config import get_project_root, load_config
from ..utils.logger import get_logger

logger = get_logger("DataDownload")


def download_raw_dataset(force: bool = False) -> Tuple[Path, pd.DataFrame]:
    """Downloads raw dataset if not present locally, or copies existing raw file to data/raw.

    Args:
        force: If True, forces re-download even if file exists.

    Returns:
        Tuple[Path, pd.DataFrame]: Path to raw file and loaded raw DataFrame.
    """
    cfg = load_config()
    root = get_project_root()

    raw_path = root / cfg["data"]["raw_path"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    # Check alternative existing location data/Telco_Customer_Churn.csv
    legacy_raw_path = root / "data" / "Telco_Customer_Churn.csv"

    if not force and raw_path.exists():
        logger.info(f"Raw dataset already exists at: {raw_path}")
    elif not force and legacy_raw_path.exists() and legacy_raw_path != raw_path:
        logger.info(f"Copying raw dataset from legacy path {legacy_raw_path} to {raw_path}")
        df_legacy = pd.read_csv(legacy_raw_path)
        df_legacy.to_csv(raw_path, index=False)
    else:
        url = cfg["data"]["source_url"]
        logger.info(f"Downloading raw dataset from: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            content = response.read().decode("utf-8")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Dataset successfully saved to: {raw_path}")

    df = pd.read_csv(raw_path)
    file_size_kb = raw_path.stat().st_size / 1024.0
    logger.info(f"Raw dataset loaded. File size: {file_size_kb:.2f} KB | Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    return raw_path, df


if __name__ == "__main__":
    download_raw_dataset()
