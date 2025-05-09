import os
import subprocess
import threading
import time

from src.gdrive_ops import check_for_new_files, download_file, sync_folder
from src.logger import setup_logger
from src.train import Trainer

logger = setup_logger()

MONITORING_INTERVAL = 10


class TrainManager:
    def __init__(
        self,
        local_dataset_path: str,
        local_model_logs_path: str,
        gdrive_dataset_path: str,
        gdrive_model_logs_path: str
    ):
        """
        Initialize the TrainManager.
        """
        self.local_dataset_path = local_dataset_path
        self.local_model_logs_path = local_model_logs_path
        self.gdrive_dataset_path = gdrive_dataset_path
        self.gdrive_model_logs_path = gdrive_model_logs_path

        self.training_queue = []
        self.is_training_process_alive = False

        self._initialize()

    def _initialize(self):
        """
        Get the current dataset paths and model logs in Google Drive.
        """
        self.current_local_dataset_paths = check_for_new_files(self.local_dataset_path, [], is_gdrive=False)
        self.current_gdrive_dataset_paths = check_for_new_files(self.gdrive_dataset_path, [])
        self.current_local_model_logs = check_for_new_files(self.local_model_logs_path, [], is_gdrive=False)
        self.current_gdrive_model_logs = check_for_new_files(self.gdrive_model_logs_path, [])

    def start(self):
        """
        Start the monitoring process.
        """
        logger.info("Starting receiving new datasets from Google Drive...")
        self._sync_model_logs_loop()

        while True:
            # step 1 check for new files in the dataset path
            new_dataset_files = check_for_new_files(self.gdrive_dataset_path, self.current_gdrive_dataset_paths)
            self.current_gdrive_dataset_paths += new_dataset_files
            self.training_queue += new_dataset_files

            if len(self.training_queue) == 0 or self.is_training_process_alive:
                if 'training_proc' in locals():
                    self.is_training_process_alive = self._check_for_alive_training_process_status(training_proc)
                    if self.is_training_process_alive:
                        logger.info(f"Training process is still running! Training queue: {self.training_queue}")
                time.sleep(MONITORING_INTERVAL)
                continue

            # step 2 if there are new files, download them
            new_dataset = self.training_queue.pop(0)
            logger.info(f"Prepare training for {new_dataset}...")
            new_dataset_path = os.path.join(self.gdrive_dataset_path, new_dataset)
            local_new_dataset_path = download_file(new_dataset_path, self.local_dataset_path, show_progress=False)

            # step 3 prepare the dataset for training in the local dataset path
            dataset_path = Trainer.prepare_dataset(local_new_dataset_path)

            # step 4 trigger the training process in a separate process
            dataset_name = os.path.basename(dataset_path)
            training_proc = self._trigger_training_process(run_name=dataset_name,
                                                           dataset_path=dataset_path,
                                                           model_log_path=self.local_model_logs_path)

            self.is_training_process_alive = self._check_for_alive_training_process_status(training_proc)
            if not self.is_training_process_alive:
                training_proc.terminate()

    def _trigger_training_process(self, run_name: str, dataset_path: str, model_log_path: str):
        """
        Trigger the training process in a separate process.
        """
        # Create a command to run the training script
        cmd = [
            "python", "src/train.py",
            "--run_name", run_name,
            "--data_path", dataset_path,
            "--model_log_path", model_log_path
        ]

        # Run the process in the background
        training_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"Started training process with PID: {training_proc.pid}")
        return training_proc

    def _check_for_alive_training_process_status(self, training_proc: subprocess.Popen):
        """
        Check if the training process is still running.
        """
        if training_proc.poll() is None:
            logger.info(f"Training process is still running (PID: {training_proc.pid})")
            return True
        return False

    def _sync_model_logs_loop(self):
        def sync_task():
            while True:
                try:
                    sync_folder(self.local_model_logs_path, self.gdrive_model_logs_path, show_progress=True)
                except Exception as e:
                    logger.error(f"Error during auto-sync: {e}")
                time.sleep(60)

        sync_thread = threading.Thread(target=sync_task, daemon=True)
        sync_thread.start()
        return sync_thread


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", type=str, required=True)
    parser.add_argument("--local_model_logs_path", type=str, required=True)
    parser.add_argument("--gdrive_dataset_path", type=str, required=True)
    parser.add_argument("--gdrive_model_logs_path", type=str, required=True)
    args = parser.parse_args()

    # ensure that the local dataset and model logs paths exist
    os.makedirs(args.local_dataset_path, exist_ok=True)
    os.makedirs(args.local_model_logs_path, exist_ok=True)

    train_manager = TrainManager(
        local_dataset_path=args.local_dataset_path,
        local_model_logs_path=args.local_model_logs_path,
        gdrive_dataset_path=args.gdrive_dataset_path,
        gdrive_model_logs_path=args.gdrive_model_logs_path
    )
    train_manager.start()