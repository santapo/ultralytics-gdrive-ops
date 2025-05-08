import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(name="ultralytics_gdrive_ops", log_dir="logs", level=logging.INFO):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, f"{name}.log")

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler = RotatingFileHandler(log_path, maxBytes=2*1024*1024, backupCount=5)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False  # Prevent duplicate logs

    return logger


def get_logger(name="ultralytics_gdrive_ops"):
    return logging.getLogger(name)