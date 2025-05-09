import argparse
import multiprocessing
import os
import time

from src.gdrive_ops import check_for_new_files, download_file, upload_file
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
        logger.info("Starting TrainerManager...")
        while True:
            # step 1 check for new files in the dataset path
            new_dataset_files = check_for_new_files(self.gdrive_dataset_path, self.current_gdrive_dataset_paths)
            self.training_queue += new_dataset_files

            if len(self.training_queue) == 0 or self.is_training_process_alive:
                time.sleep(MONITORING_INTERVAL)
                continue

            # step 2 if there are new files, download them
            new_dataset = self.training_queue.pop(0)
            new_dataset_path = os.path.join(self.gdrive_dataset_path, new_dataset)
            local_new_dataset_path = download_file(new_dataset_path, self.local_dataset_path)

            # step 3 prepare the dataset for training in the local dataset path
            dataset_path = Trainer.prepare_dataset(local_new_dataset_path)

            # step 4 trigger the training process in a separate process
            training_proc = self._trigger_training_process(dataset_path)

            self.is_training_process_alive = self._check_for_alive_training_process_status(training_proc)

            # step 5 sync the model logs
            # self._sync_model_logs()

    def _trigger_training_process(self, dataset_path: str):
        """
        Trigger the training process in a separate process.
        """
        def train_process():
            try:
                # trainer = Trainer(model_path="yolo11n.pt", data_path=dataset_path)
                # trainer.train()
                print("start training")
                for i in range(20):
                    print(f"training {i}...")
                    time.sleep(1)
                print("training completed")
                logger.info("Training completed successfully")
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Training failed with error: {str(e)}")

        training_proc = multiprocessing.Process(target=train_process)
        training_proc.start()
        return training_proc

    def _check_for_alive_training_process_status(self, training_proc: multiprocessing.Process):
        """
        Check if the training process is still running.
        """
        if training_proc.is_alive():
            logger.info(f"Training process is still running (PID: {training_proc.pid})")
            return True
        return False

    def _sync_model_logs(self):
        """
        Sync the model logs to the Google Drive model logs path.
        """
        new_model_logs = check_for_new_files(self.local_model_logs_path, self.current_gdrive_model_logs)
        if len(new_model_logs) == 0:
            time.sleep(MONITORING_INTERVAL)
            return

        for model_log in new_model_logs:
            upload_file(model_log, self.gdrive_model_logs_path)


if __name__ == "__main__":
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