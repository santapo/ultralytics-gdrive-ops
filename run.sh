#!/bin/bash


python src/main.py \
    --local_dataset_path ./logs/datasets \
    --local_model_logs_path ./logs/training_logs \
    --local_pretrained_model_path ./logs/pretrained_models \
    --gdrive_dataset_path po_gdrive:regent-fabric-training-repo/datasets \
    --gdrive_model_logs_path po_gdrive:regent-fabric-training-repo/training_logs \
    --gdrive_pretrained_model_path po_gdrive:regent-fabric-training-repo/pretrained_models