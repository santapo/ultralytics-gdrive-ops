import os
from copy import deepcopy
from datetime import datetime
from functools import partial
from typing import Optional

import torch
import yaml
from ultralytics import YOLO
from ultralytics.yolo.utils import __version__
from ultralytics.yolo.utils.torch_utils import de_parallel

from src.gdrive_ops import download_file
from src.logger import get_logger
from src.utils import check_dataset_structure, unzip_file

logger = get_logger("ultralytics_gdrive_ops")


class Trainer:
    def __init__(
        self,
        run_name: str,
        data_path: str,
        model_log_path: str,
        local_pretrained_model_path: str,
        gdrive_pretrained_model_path: str
    ):
        self.run_name = run_name
        self.data_path = data_path
        self.model_log_path = model_log_path
        self.local_pretrained_model_path = local_pretrained_model_path
        self.gdrive_pretrained_model_path = gdrive_pretrained_model_path

    @staticmethod
    def prepare_dataset(dataset_path: str) -> str | None:
        """
        Prepare the dataset for training.
        """
        if not dataset_path.endswith('.zip'):
            logger.error(f"Dataset path must end with .zip: {dataset_path}")
            return None

        save_dir = os.path.dirname(dataset_path)
        unzip_file(dataset_path, save_dir)

        dataset_path = os.path.splitext(dataset_path)[0]

        if not check_dataset_structure(dataset_path):
            return None

        return dataset_path

    def _prepare_data_yaml(self, config_file: str):
        """
        Prepare the data.yaml file. If it doesn't exist, create a default one.
        """
        if not os.path.exists(config_file):
            data_yaml = {
                "path": os.path.join(os.getcwd(), self.data_path),
                "train": "train/images",
                "val": "val/images",
                "names": {
                    0: "defect",
                }
            }
        else:
            with open(config_file, "r") as f:
                data_yaml = yaml.safe_load(f)
            data_yaml["path"] = os.path.join(os.getcwd(), self.data_path)

        with open(config_file, "w") as f:
            yaml.dump(data_yaml, f)

    def _prepare_model(self, pretrained_model: Optional[str]):
        """
        Prepare the model. If the pretrained model is not provided, use the default one.
        """
        if not pretrained_model:
            pretrained_model = "yolov8x.pt"

        if not pretrained_model.startswith("yolo"):
            source_path = os.path.join(self.gdrive_pretrained_model_path, pretrained_model)
            pretrained_model = download_file(source_path, self.local_pretrained_model_path)

        logger.info(f"Loading pretrained model from {pretrained_model}...")
        model = YOLO(pretrained_model)
        return model

    def _prepare_training_args(self, training_config_file: Optional[str] = None):
        """
        Prepare the training arguments. If the training config file is not provided, use the default one.
        """
        default_training_args = {
            "pretrained_model": "yolov8x.pt",
            "epochs": 300,
            "imgsz": 960,
            "batch": 8,
            "exist_ok": True,
            "save_period": -1,
            "fliplr": 0.8,
            "flipud": 0.6,
            "scale": 0.1,
            "patience": 200
        }

        if training_config_file:
            with open(training_config_file, "r") as f:
                training_args = yaml.safe_load(f)
        else:
            training_args = default_training_args

        training_args["project"] = os.path.join(os.getcwd(), self.model_log_path)
        training_args["name"] = self.run_name

        return training_args

    def train(self):
        """
        Train the model.
        """
        training_config_file = os.path.join(self.data_path, "training_config.yaml")
        training_args = self._prepare_training_args(training_config_file)

        pretrained_model = training_args.pop("pretrained_model")
        model = self._prepare_model(pretrained_model=pretrained_model)

        data_config_file = os.path.join(self.data_path, "data.yaml")
        self._prepare_data_yaml(data_config_file)


        save_top10_models_callback = partial(save_topk_models_callback, k=10)
        model.add_callback("on_fit_epoch_end", save_top10_models_callback)
        model.train(
            data=data_config_file,
            **training_args
        )

        model.export(format='onnx', opset=12, dynamic= True)


import heapq


def save_topk_models_callback(trainer, k=10):
    """
    Callback to save top k models based on fitness score.

    Args:
        trainer: The trainer object
        k: Number of top models to keep (default: 10)
    """
    # Initialize topk heap on first call
    if not hasattr(trainer, "_topk_models"):
        trainer._topk_models = []  # min-heap of (fitness, epoch, path)
        trainer._topk_ckpt_paths = set()

    # Prepare checkpoint
    ckpt = {
        'epoch': trainer.epoch,
        'best_fitness': trainer.best_fitness,
        'model': deepcopy(de_parallel(trainer.model)).half(),
        'ema': deepcopy(trainer.ema.ema).half(),
        'updates': trainer.ema.updates,
        'optimizer': trainer.optimizer.state_dict(),
        'train_args': vars(trainer.args),  # save as dict
        'date': datetime.now().isoformat(),
        'version': __version__
    }

    map_score = trainer.metrics["metrics/mAP50-95(B)"]
    epoch = trainer.epoch
    topk = trainer._topk_models
    ckpt_name = f"top{k}_epoch{epoch}_map5095_{map_score:.6f}.pt"
    ckpt_path = trainer.wdir / ckpt_name

    # If less than k, just add
    if len(topk) < k:
        heapq.heappush(topk, (map_score, epoch, str(ckpt_path)))
        trainer._topk_ckpt_paths.add(str(ckpt_path))
        torch.save(ckpt, ckpt_path)
    else:
        # If current score is better than the worst in topk, replace
        if map_score > topk[0][0]:
            # Remove the worst
            _, _, worst_path = heapq.heappop(topk)
            if worst_path in trainer._topk_ckpt_paths:
                try:
                    os.remove(worst_path)
                except Exception:
                    pass
                trainer._topk_ckpt_paths.remove(worst_path)
            # Add the new one
            heapq.heappush(topk, (map_score, epoch, str(ckpt_path)))
            trainer._topk_ckpt_paths.add(str(ckpt_path))
            torch.save(ckpt, ckpt_path)

    del ckpt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_log_path", type=str, required=True)
    parser.add_argument("--local_pretrained_model_path", type=str, required=True)
    parser.add_argument("--gdrive_pretrained_model_path", type=str, required=True)
    args = parser.parse_args()

    trainer = Trainer(
        run_name=args.run_name,
        data_path=args.data_path,
        model_log_path=args.model_log_path,
        local_pretrained_model_path=args.local_pretrained_model_path,
        gdrive_pretrained_model_path=args.gdrive_pretrained_model_path
    )
    trainer.train()
