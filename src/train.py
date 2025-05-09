import os

import yaml
from ultralytics import YOLO

from src.logger import get_logger
from src.utils import check_dataset_structure, unzip_file

logger = get_logger("ultralytics_gdrive_ops")


class Trainer:
    def __init__(self, run_name: str, data_path: str, model_log_path: str):
        self.run_name = run_name
        self.data_path = data_path
        self.model_log_path = model_log_path

    @staticmethod
    def prepare_dataset(dataset_path: str) -> str | None:
        """
        Prepare the dataset for training.
        """
        if not dataset_path.endswith('.zip'):
            logger.error(f"Dataset path must end with .zip: {dataset_path}")
            return None

        save_dir = os.path.splitext(dataset_path)[0]
        unzip_dir = unzip_file(dataset_path, save_dir)

        if unzip_dir is None:
            return None

        if not check_dataset_structure(unzip_dir):
            return None

        return unzip_dir

    def train(self):
        """
        Train the model.
        """
        model = YOLO("yolov8x.pt")

        config_file = os.path.join(self.data_path, "data.yaml")

        # replace the path in the config file with absolute path
        with open(config_file, "r") as f:
            data_yaml = yaml.safe_load(f)

        data_yaml["path"] = os.path.join(os.getcwd(), self.data_path)

        with open(config_file, "w") as f:
            yaml.dump(data_yaml, f)

        train_results = model.train(
            data=config_file,
            epochs=2,
            imgsz=1280,
            batch=8,
            name=self.run_name,
            project=os.path.join(os.getcwd(), self.model_log_path),
            exist_ok=True,
            save_period=1,
            fliplr=0.8,
            flipud=0.6,
            scale=0.1,
            patience=200
        )

        model.export(format="onnx", imgsz=1280)