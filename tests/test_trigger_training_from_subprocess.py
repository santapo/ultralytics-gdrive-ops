import os
import subprocess
import time


def test_trigger_training_from_subprocess():
    env = os.environ.copy()
    cmd = [
        "python", "src/train.py",
        "--run_name", "test",
        "--data_path", "logs/datasets/data_test",
        "--model_log_path", "logs/training_logs/",
        "--local_pretrained_model_path", "logs/pretrained_models/",
        "--gdrive_pretrained_model_path", "po_gdrive:regent-fabric-training-repo/pretrained_models"
    ]

    env["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        training_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
    except Exception as e:
        print(e)

    while training_proc.poll() is None:
        print("Training process is running...")
        time.sleep(1)


test_trigger_training_from_subprocess()
