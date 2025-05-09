import os
import zipfile

from src.logger import get_logger

logger = get_logger("ultralytics_gdrive_ops")


def unzip_file(file_path: str, extract_dir: str):
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    except Exception as e:
        logger.error(f"Error unzipping file {file_path}: {e}")


def check_dataset_structure(dataset_path: str) -> bool:
    """
    Verify the structure of the dataset.
    """
    def check_subset_structure(subset_path: str) -> bool:
        """
        Check the structure of a subset of the dataset.
        """
        if not os.path.exists(subset_path) or not os.path.isdir(subset_path):
            logger.error(f"Subset directory not found at {subset_path}")
            return False

        images_dir = os.path.join(subset_path, "images")
        labels_dir = os.path.join(subset_path, "labels")

        if not os.path.exists(images_dir) or not os.path.isdir(images_dir):
            logger.error(f"Images directory not found at {images_dir}")
            return False

        if not os.path.exists(labels_dir) or not os.path.isdir(labels_dir):
            logger.error(f"Labels directory not found at {labels_dir}")
            return False

        return True

    # Check if the dataset has the required train and validation directories
    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "val")

    if not os.path.exists(train_dir) or not os.path.isdir(train_dir):
        logger.error(f"Training directory not found at {train_dir}")
        return False

    if not os.path.exists(val_dir) or not os.path.isdir(val_dir):
        logger.error(f"Validation directory not found at {val_dir}")
        return False

    is_train_valid = check_subset_structure(train_dir)
    if not is_train_valid:
        logger.error(f"Training directory structure is invalid: {train_dir}")
        return False

    is_val_valid = check_subset_structure(val_dir)
    if not is_val_valid:
        logger.error(f"Validation directory structure is invalid: {val_dir}")
        return False

    logger.info(f"Dataset structure verified: {train_dir} and {val_dir}")
    return True