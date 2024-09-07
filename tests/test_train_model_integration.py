import pytest
from src.config import BoldOrNotConfig
from src.model_training import train_model


def test_num_of_epochs(test_config: BoldOrNotConfig) -> None:
    history = train_model(config=test_config)
    assert len(history.epoch) == test_config.training_params.epochs
