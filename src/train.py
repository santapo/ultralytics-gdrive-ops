import os

from ultralytics import YOLO

from src.logger import get_logger
from src.utils import check_dataset_structure, unzip_file

logger = get_logger("ultralytics_gdrive_ops")


class Trainer:
    def __init__(self, run_name: str, model_path: str, data_path: str):
        self.run_name = run_name
        self.model_path = model_path
        self.data_path = data_path

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
        model = YOLO("yolo11n.pt")
        model.train(data="data.yaml", epochs=100)

        train_results = model.train(
            data="coco8.yaml",  # Path to dataset configuration file
            epochs=100,  # Number of training epochs
            imgsz=640,  # Image size for training
            device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
        )

        metrics = model.val()

        path = model.export(format="onnx")  # Returns the path to the exported model

        ...