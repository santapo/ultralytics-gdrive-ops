


from src.train import Trainer


def test_trainer():
    # dataset_path = Trainer.prepare_dataset("./tmp/data_train_v4.zip")
    dataset_path = "./logs/datasets/data_train_v5_8x"
    print(dataset_path)
    trainer = Trainer(
        "test",
        dataset_path,
        "logs/training_logs",
        local_pretrained_model_path="logs/pretrained_models",
        gdrive_pretrained_model_path="po_gdrive:regent-fabric-training-repo/pretrained_models"
    )
    trainer.train()

test_trainer()