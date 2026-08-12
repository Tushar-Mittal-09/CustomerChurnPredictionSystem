"""
Phase 1 Smoke Tests: Configuration, Logger, and Project Setup.
"""
from pathlib import Path
import pytest
from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger


def test_project_root_exists():
    """Verify that project root directory is identified correctly."""
    root = get_project_root()
    assert root.exists()
    assert (root / "configs" / "config.yaml").exists()


def test_load_config():
    """Verify that config.yaml loads valid YAML settings."""
    cfg = load_config()
    assert "project" in cfg
    assert cfg["project"]["name"] == "ChurnGuard AI"
    assert "data" in cfg
    assert "target_column" in cfg["data"]
    assert cfg["data"]["target_column"] == "Churn"


def test_logger_initialization(tmp_path):
    """Verify that get_logger creates console and file handlers correctly."""
    log_file = tmp_path / "test.log"
    logger = get_logger("TestLogger", log_file=str(log_file))
    logger.info("Test log message")
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Test log message" in content
