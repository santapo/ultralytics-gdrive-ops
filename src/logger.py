import logging
import os
from logging.handlers import RotatingFileHandler


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[91m\033[1m', # Bold Red
        'RESET': '\033[0m'    # Reset
    }

    def format(self, record):
        log_message = super().format(record)
        level_name = record.levelname
        if level_name in self.COLORS:
            return f"{self.COLORS[level_name]}{log_message}{self.COLORS['RESET']}"
        return log_message


def setup_logger(name="ultralytics_gdrive_ops", log_dir="logs", level=logging.INFO):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, f"{name}.log")

    file_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_formatter = ColoredFormatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # File handler
    handler = RotatingFileHandler(log_path, maxBytes=2*1024*1024, backupCount=5)
    handler.setFormatter(file_formatter)
    handler.setLevel(level)
    logger.addHandler(handler)

    # Console handler with colored output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    logger.propagate = False  # Prevent duplicate logs

    return logger


def get_logger(name="ultralytics_gdrive_ops"):
    return logging.getLogger(name)