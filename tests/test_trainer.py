


from src.train import Trainer


def test_trainer():
    dataset_path = Trainer.prepare_dataset("./tmp/yafter_test1_yolo.zip")
    print(dataset_path)
    trainer = Trainer("test", dataset_path, "logs/training_logs")
    trainer.train()

test_trainer()