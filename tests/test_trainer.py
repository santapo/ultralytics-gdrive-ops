


from src.train import Trainer


def test_trainer():
    trainer = Trainer("test", "tmp/yafter_test1_yolo/", "logs/training_logs")
    trainer.train()

test_trainer()