from unittest.mock import patch, MagicMock
from src.model_training import train_model

@patch('src.model_training.BaldOrNotModel.fit')
@patch('src.model_training.BaldDataset')
def test_train_model(MockBaldDataset, MockFit, test_config):
    MockFit.return_value = MagicMock()

    history = train_model(test_config)

    MockFit.assert_called_once()
    assert history is not None
