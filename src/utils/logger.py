"""
Centralized Logging Utility for ChurnGuard AI.
"""
import logging
from pathlib import Path
import sys
from typing import Optional
from .config import get_project_root, load_config


def get_logger(name: str = "ChurnGuardAI", log_file: Optional[str] = None, level: Optional[str] = None) -> logging.Logger:
    """Configures and returns a standardized logger instance.

    Args:
        name: Name of the logger.
        log_file: Optional filepath for file logging.
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    try:
        cfg = load_config()
        log_cfg = cfg.get("logging", {})
        default_level = log_cfg.get("level", "INFO")
        default_file = log_cfg.get("file", "logs/project.log")
    except Exception:
        default_level = "INFO"
        default_file = "logs/project.log"

    selected_level = getattr(logging, (level or default_level).upper(), logging.INFO)
    logger.setLevel(selected_level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    target_log_file = log_file or default_file
    root = get_project_root()
    full_log_path = root / target_log_file
    full_log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(full_log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
