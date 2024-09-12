import os
import shutil
from jsonargparse import CLI
import json
from dataclasses import asdict
from src.config_class import BoldOrNotConfig
from src.logging import setup_logging
from src.model_training import train_model, init_output_dir
from src.plot import plot_metric_curve


def run_experiment(config: BoldOrNotConfig):
    output_dir_path = init_output_dir()
    setup_logging(output_dir_path)
    config_path = os.path.join(output_dir_path, "config.yaml")
    shutil.copy("../config.yaml", config_path)

    print("Running experiment with the following configuration:")
    print(json.dumps(asdict(config), indent=4))
    print("Start training:")
    history = train_model(config=config, output_dir_path=output_dir_path)
    plot_metric_curve(
        history=history, metric="loss", output_dir_path=output_dir_path
    )
    plot_metric_curve(
        history=history, metric="accuracy", output_dir_path=output_dir_path
    )


if __name__ == "__main__":
    config = CLI(BoldOrNotConfig)
    run_experiment(config=config)
