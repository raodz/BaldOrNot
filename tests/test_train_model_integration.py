import os.path
import shutil
from src.config import BoldOrNotConfig
from src.model_training import train_model


def test_num_of_epochs(test_config: BoldOrNotConfig) -> None:
    history = train_model(config=test_config)
    assert len(history.epoch) == test_config.training_params.epochs, (
        "There is difference between number of epochs in config "
        "and number of epochs in history"
    )


def test_tensorboard_logs_saving(test_config: BoldOrNotConfig) -> None:
    log_dir = "tensorboard_logs_test"

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    train_model(config=test_config)

    assert os.path.exists(log_dir), "Log directory was not created"

    log_files = os.listdir(log_dir)
    assert len(log_files) > 0, "Log directory is empty"

    shutil.rmtree(log_dir)


def test_early_stopping(test_config: BoldOrNotConfig) -> None:
    history = train_model(config=test_config)
    early_stopping_patience = 3
    val_loss_history = history.history['val_loss']
    best_epoch = val_loss_history.index(min(val_loss_history))
    num_epochs_trained = len(history.epoch)

    assert num_epochs_trained - best_epoch - 1 == early_stopping_patience
