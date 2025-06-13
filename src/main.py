import os
import subprocess
import sys
import threading
import time
from typing import Optional

from src.gdrive_ops import check_for_new_files, download_file, sync_folder
from src.logger import setup_logger
from src.train import Trainer

logger = setup_logger("ultralytics_gdrive_ops")

MONITORING_INTERVAL = 10
MAX_SIMULTANEOUS_TRAINING_PROCESSES = 2


class TrainManager:
    def __init__(
        self,
        local_dataset_path: str,
        local_model_logs_path: str,
        local_pretrained_model_path: str,
        gdrive_dataset_path: str,
        gdrive_model_logs_path: str,
        gdrive_pretrained_model_path: str
    ):
        """
        Initialize the TrainManager.
        """
        self.local_dataset_path = local_dataset_path
        self.local_model_logs_path = local_model_logs_path
        self.local_pretrained_model_path = local_pretrained_model_path
        self.gdrive_dataset_path = gdrive_dataset_path
        self.gdrive_model_logs_path = gdrive_model_logs_path
        self.gdrive_pretrained_model_path = gdrive_pretrained_model_path

        self.training_queue = []
        self.stop_sync_thread = False
        self.inuse_gpu_ids = []
        self.training_procs = []

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
        sync_thread = self._sync_model_logs_loop()

        try:
            while True:
                # Check for new datasets
                self._check_and_update_datasets()

                # Check for alive training processes
                self._check_for_alive_training_processes()

                # Start new training if queue has items
                if not self.training_queue:
                    logger.info(f"No new datasets to train. Training queue: {self.training_queue}")
                    time.sleep(MONITORING_INTERVAL)
                    continue

                if len(self.training_procs) >= MAX_SIMULTANEOUS_TRAINING_PROCESSES:
                    logger.info(f"Max simultaneous training processes reached. Training queue: {self.training_queue}")
                    time.sleep(MONITORING_INTERVAL)
                    continue

                # Process next dataset in queue
                self._process_next_dataset()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Exiting...")
            self._cleanup(sync_thread)
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error during training: {e}")
            self._cleanup(sync_thread)
            sys.exit(1)

    def _check_and_update_datasets(self):
        """Check for new datasets and add them to the training queue."""
        new_dataset_files = check_for_new_files(self.gdrive_dataset_path, self.current_gdrive_dataset_paths)
        self.current_gdrive_dataset_paths += new_dataset_files
        self.training_queue += new_dataset_files

    def _process_next_dataset(self):
        """Process the next dataset in the queue and start training."""
        gpu_id = self._get_free_gpu_id(self.inuse_gpu_ids)
        if gpu_id is None:
            logger.warning("No free GPU found, skipping training")
            return

        new_dataset = self.training_queue.pop(0)
        logger.info(f"Prepare training for {new_dataset}...")

        # Download the dataset
        new_dataset_path = os.path.join(self.gdrive_dataset_path, new_dataset)
        local_new_dataset_path = download_file(new_dataset_path, self.local_dataset_path, show_progress=False)

        # Prepare the dataset for training
        dataset_path = Trainer.prepare_dataset(local_new_dataset_path)

        # Start the training process
        dataset_name = os.path.basename(dataset_path)
        self._trigger_training_process(
            run_name=dataset_name,
            dataset_path=dataset_path,
            model_log_path=self.local_model_logs_path,
            local_pretrained_model_path=self.local_pretrained_model_path,
            gdrive_pretrained_model_path=self.gdrive_pretrained_model_path,
            gpu_id=gpu_id
        )

    def _cleanup(self, sync_thread: threading.Thread):
        """Clean up resources before exiting."""
        if sync_thread:
            self.stop_sync_thread = True
            sync_thread.join(timeout=2)
        for training_proc in self.training_procs:
            training_proc.terminate()
            self.training_procs.remove(training_proc)
            self.inuse_gpu_ids.remove(training_proc.gpu_id)

    def _trigger_training_process(
            self,
            run_name: str,
            dataset_path: str,
            model_log_path: str,
            local_pretrained_model_path: str,
            gdrive_pretrained_model_path: str,
            gpu_id: int
        ):
        """
        Trigger the training process in a separate process.
        """
        # Create a command to run the training script
        cmd = [
            "CUDA_VISIBLE_DEVICES=" + str(gpu_id),
            "python", "src/train.py",
            "--run_name", run_name,
            "--data_path", dataset_path,
            "--model_log_path", model_log_path,
            "--local_pretrained_model_path", local_pretrained_model_path,
            "--gdrive_pretrained_model_path", gdrive_pretrained_model_path
        ]
        self.inuse_gpu_ids.append(gpu_id)

        env = os.environ.copy()
        if "WANDB_API_KEY" in os.environ:
            env["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

        try:
            training_proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.training_procs.append(training_proc)
        except Exception as e:
            logger.error(f"Error during training: {e}")

    def _check_for_alive_training_processes(self):
        """
        Check if the training process is still running.
        """
        for training_proc in self.training_procs:
            if training_proc.poll() is None:
                logger.info(f"Training process is still running (PID: {training_proc.pid})")
            else:
                self.training_procs.remove(training_proc)
                self.inuse_gpu_ids.remove(training_proc.gpu_id)

    def _sync_model_logs_loop(self):
        def sync_task():
            while not self.stop_sync_thread:
                try:
                    sync_folder(self.local_model_logs_path, self.gdrive_model_logs_path, show_progress=True)
                except Exception as e:
                    logger.error(f"Error during auto-sync: {e}")

                # wait for 60 seconds, if the sync thread is stopped, break
                for _ in range(1200):
                    if self.stop_sync_thread:
                        break
                    time.sleep(1)

        self.stop_sync_thread = False   # ensure that the sync thread is not stopped
        sync_thread = threading.Thread(target=sync_task, daemon=False)
        sync_thread.start()
        return sync_thread

    @staticmethod
    def _get_free_gpu_id(inuse_gpu_ids: list[int]) -> Optional[int]:
        """
        Get the free GPU id.
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True
            )

            memory_usage = [int(x) for x in result.stdout.strip().split('\n')]
            for gpu_id, usage in enumerate(memory_usage):
                if usage <= 20 and gpu_id not in inuse_gpu_ids:
                    return gpu_id

            logger.warning("No GPU found with usage <= 20MB")
        except Exception as e:
            logger.error(f"Error getting free GPU: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", type=str, required=True)
    parser.add_argument("--local_model_logs_path", type=str, required=True)
    parser.add_argument("--local_pretrained_model_path", type=str, required=True)
    parser.add_argument("--gdrive_dataset_path", type=str, required=True)
    parser.add_argument("--gdrive_model_logs_path", type=str, required=True)
    parser.add_argument("--gdrive_pretrained_model_path", type=str, required=True)
    args = parser.parse_args()

    # ensure that the local dataset and model logs paths exist
    os.makedirs(args.local_dataset_path, exist_ok=True)
    os.makedirs(args.local_model_logs_path, exist_ok=True)

    train_manager = TrainManager(
        local_dataset_path=args.local_dataset_path,
        local_model_logs_path=args.local_model_logs_path,
        local_pretrained_model_path=args.local_pretrained_model_path,
        gdrive_dataset_path=args.gdrive_dataset_path,
        gdrive_model_logs_path=args.gdrive_model_logs_path,
        gdrive_pretrained_model_path=args.gdrive_pretrained_model_path
    )
    train_manager.start()