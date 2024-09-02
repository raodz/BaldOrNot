from jsonargparse import CLI
import json
from dataclasses import asdict
from src.config import BoldOrNotConfig
from src.model_training import train_model


def run_experiment(config: BoldOrNotConfig):
    print("Running experiment with the following configuration:")
    print(json.dumps(asdict(config), indent=4))
    print('Start training:')
    train_model(config=config)


if __name__ == "__main__":
    config = CLI(BoldOrNotConfig)
    run_experiment(config=config)
