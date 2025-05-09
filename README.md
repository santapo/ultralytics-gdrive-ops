

# Ultralytics Google Drive Training Manager

This service automates the process of training Ultralytics YOLOv8 models using datasets stored in Google Drive. It continuously monitors a specified Google Drive folder for new datasets, downloads them, initiates training, and syncs the training results back to Google Drive.

![Ultralytics Google Drive Training Manager](assets/arch.jpg)

## How It Works

1. **Monitoring**: Continuously checks a designated Google Drive folder for new datasets
2. **Download**: When a new dataset is detected, it's downloaded to the local machine
3. **Training**: Automatically starts YOLOv8 training on the downloaded dataset
4. **Synchronization**: Training results and model logs are synced back to Google Drive in real-time

Note that it runs solely on a single GPU instance and only triggers one training at a time.

## Prerequisites

- Python 3.10+
- rclone configured with Google Drive access
- Ultralytics YOLOv8 installed
- ONNX in order to export trained models to ONNX format

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ultralytics-gdrive-trainer.git
   cd ultralytics-gdrive-trainer
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure rclone for Google Drive access:
   ```bash
   rclone config
   ```

## Usage

Run the training manager with the following command:
```
bash run.sh
```