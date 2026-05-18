import logging
import sys
from pathlib import Path


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """Configure structured, low-noise logging for API and batch processing."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    file_handler = logging.FileHandler(Path(log_dir) / "roadvision.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)
