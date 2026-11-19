# utils/logging_config.py

"""
Centralized logging configuration for the RAG system.

Engineering principle: Configure logging ONCE at startup, use it everywhere.

Why centralize?
    If every module sets up its own logging configuration, they'll conflict.
    You'll get duplicate messages, inconsistent formats, and chaos.
    One place sets the rules. Every module just says "give me a logger" and uses it.

Usage in any module:
    from utils.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened", extra={"key": "value"})
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# LOG FORMAT

CONSOLE_FORMAT = (
    "%(asctime)s.%(msecs)03d | "   # Timestamp with milliseconds
    "%(levelname)-8s | "           # Level (padded to 8 chars)
    "%(name)-25s | "               # Module name (padded to 25 chars)
    "%(message)s"                  # The actual message
)

# File logs get more detail: thread name helps debug multithreading issues
FILE_FORMAT = (
    "%(asctime)s.%(msecs)03d | "
    "%(levelname)-8s | "
    "%(name)-25s | "
    "%(threadName)-15s | "         # Which thread produced this log
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# SETUP FUNCTION -- to be called once at startup in main.py

def setup_logging(
        console_level: int = logging.INFO,
        log_to_file: bool = True,
        log_dir: str = "logs",
        ) -> None:
    """
    Set up logging configuration for the entire application.
    Called once at the start of the program (e.g., in main.py).
    
    Args:
        console_level: Logging level for console output (default: INFO).
        log_to_file: Whether to log to a file (default: True).
        log_dir: Directory where log files will be stored (default: "logs").
    """
    # Create a root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    root_logger.handlers.clear()  # Clear any existing handlers

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File Handler 
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)  # Create log directory if it doesn't exist

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"rag_log_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_formatter = logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"File logging enabled. Logs will be written to: {log_file}")
    _silence_noisy_libraries()

def _silence_noisy_libraries() -> None:
    """
    Silence overly verbose loggers from third-party libraries.
    Adjust levels as needed for your application.
    """
    noisy_loggers = [
        "urllib3",
        "requests",
        "botocore",
        "boto3",
        "transformers",
        "tensorflow",
        "torch",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

# GET LOGGER — call this in every module
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.

    Call at the TOP of every module file, like this:
        logger = get_logger(__name__)

    Args:
        name: The name of the logger (typically __name__).

    Returns:
        A logging.Logger instance.
    """
    return logging.getLogger(name)