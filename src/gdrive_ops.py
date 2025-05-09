import os

from rclone_python import rclone

from src.logger import get_logger

logger = get_logger("ultralytics_gdrive_ops")


def check_for_new_files(path: str, current_files: list[str], is_gdrive: bool = True) -> list[str]:
    """
    Check for new files in the Google Drive directory based on the file names.
    """
    logger.info(f"Checking for new files in {path}")
    if is_gdrive:
        all_gdrive_files = rclone.ls(path)
    else:
        all_gdrive_files = os.listdir(path)

    new_files = []
    for file in all_gdrive_files:
        if is_gdrive:
            filename = file['Path']
        else:
            filename = file
        if filename not in current_files:
            new_files.append(filename)
    logger.info(f"Found {len(new_files)} new files in {path}: {new_files}")
    return new_files


def download_file(gdrive_path: str, local_path: str):
    """
    Download a file from Google Drive to the local directory.
    """
    rclone.copy(gdrive_path, local_path, ignore_existing=False)


def upload_file(local_path: str, gdrive_path: str):
    """
    Upload a file to Google Drive.
    """
    rclone.copy(local_path, gdrive_path, ignore_existing=False)


def sync_folder(source_path: str, target_path: str):
    """
    Sync a folder to Google Drive.
    """
    rclone.sync(source_path, target_path)