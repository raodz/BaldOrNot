import os
from datetime import datetime
import shutil
from jsonargparse import CLI
import json
from dataclasses import asdict
from src.config import BoldOrNotConfig
from src.logging import setup_logging
from src.model_training import train_model
from src.plot import plot_training_curves

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
current_training = f"training{current_date}"
training_dir = os.path.join(project_path, "trainings", current_training)
os.makedirs(training_dir, exist_ok=True)
shutil.copy("../config.yaml", os.path.join(training_dir, "config.yaml"))


def run_experiment(config: BoldOrNotConfig):
    setup_logging(training_dir)

    print("Running experiment with the following configuration:")
    print(json.dumps(asdict(config), indent=4))
    print("Start training:")
    history = train_model(config=config, training_dir=training_dir)
    plot_training_curves(history, training_dir)


if __name__ == "__main__":
    config = CLI(BoldOrNotConfig)
    run_experiment(config=config)
