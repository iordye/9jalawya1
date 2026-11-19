from utils.logging_config import setup_logging, get_logger
from pathlib import Path
import logging
import pytest


@pytest.fixture(autouse=True)
def clean_root_logger():
    """
    Preserve the existing root logger configuration
    and restore it after each test.
    """
    root_logger = logging.getLogger()

    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level

    yield

    # Close handlers created during the test.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Restore original configuration.
    root_logger.handlers = original_handlers
    root_logger.setLevel(original_level)


def test_root_logger_level():
    """Root logger should be configured at DEBUG level."""
    setup_logging(log_to_file=False)

    root_logger = logging.getLogger()

    assert root_logger.level == logging.DEBUG


def test_console_logging(capsys):
    """Console handler should output log messages."""
    setup_logging(
        console_level=logging.INFO,
        log_to_file=False
    )

    logger = logging.getLogger("test")

    logger.info("Test console message")

    captured = capsys.readouterr()

    assert "Test console message" in captured.out


def test_console_respects_log_level(capsys):
    """Console should ignore messages below its configured level."""
    setup_logging(
        console_level=logging.ERROR,
        log_to_file=False
    )

    logger = logging.getLogger("test")

    logger.info("This should not appear")
    logger.error("This should appear")

    captured = capsys.readouterr()

    assert "This should not appear" not in captured.out
    assert "This should appear" in captured.out


def test_file_logging_creates_directory(tmp_path):
    """File logging should create the log directory and log file."""
    log_dir = tmp_path / "application_logs"

    setup_logging(
        console_level=logging.ERROR,
        log_to_file=True,
        log_dir=str(log_dir)
    )

    logger = logging.getLogger("test")

    logger.debug("Debug file message")
    logger.info("Info file message")

    # Flush handlers to ensure messages are written.
    for handler in logging.getLogger().handlers:
        handler.flush()

    log_files = list(log_dir.glob("rag_log_*.log"))

    assert log_dir.exists()
    assert len(log_files) == 1

    content = log_files[0].read_text()

    assert "Debug file message" in content
    assert "Info file message" in content


def test_file_logging_can_be_disabled(tmp_path):
    """Disabling file logging should prevent file creation."""
    log_dir = tmp_path / "disabled_logs"

    setup_logging(
        log_to_file=False,
        log_dir=str(log_dir)
    )

    root_logger = logging.getLogger()

    # Only the console handler should exist.
    assert len(root_logger.handlers) == 1
    assert isinstance(
        root_logger.handlers[0],
        logging.StreamHandler
    )

    assert not log_dir.exists()


def test_setup_logging_does_not_duplicate_handlers(tmp_path):
    """Repeated setup should not accumulate handlers."""
    log_dir = tmp_path / "duplicate_test"

    setup_logging(
        log_to_file=True,
        log_dir=str(log_dir)
    )

    root_logger = logging.getLogger()

    first_handlers = root_logger.handlers[:]

    assert len(first_handlers) == 2

    setup_logging(
        log_to_file=True,
        log_dir=str(log_dir)
    )

    second_handlers = root_logger.handlers[:]

    assert len(second_handlers) == 2

    # Ensure the old handlers are no longer attached.
    for handler in first_handlers:
        assert handler not in second_handlers

        # The setup function detaches but doesn't close old handlers.
        handler.close()

def test_get_logger_returns_same_instance():
    """get_logger should return the same logger instance for the same name."""
    logger1 = get_logger("test_logger")
    logger2 = get_logger("test_logger")

    assert logger1 is logger2