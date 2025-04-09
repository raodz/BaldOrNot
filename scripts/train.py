import json
import logging
import os
import shutil
from dataclasses import asdict

from jsonargparse import CLI

from src import BaldOrNotConfig
from src import train_model
from src import init_output_dir
from src import plot_metric_curve
from src.logging import setup_logging
from utils import save_metrics_report


def run_experiment(config: BaldOrNotConfig):
    logging.info("Initializing output directory...")
    output_dir_path = init_output_dir(config.training_params.training_name)

    logging.info("Setting up logging...")
    setup_logging(output_dir_path)

    config_path = os.path.join(output_dir_path, "config.yaml")
    logging.info(
        f"Copying configuration file to output directory: {config_path}"
    )
    shutil.copy("../config.yaml", config_path)

    logging.info(
        f"Running experiment with the following configuration: \n"
        f"{json.dumps(asdict(config), indent=4)}"
    )

    logging.info("Starting model training...")
    history = train_model(config=config, output_dir_path=output_dir_path)

    logging.info(
        "Model training completed. History of training:\n"
        f"{json.dumps(history.history, indent=4)}"
    )
    metrics_report = {
        "Model": {
            metric: history.history[metric][-1] for metric in history.history
        }
    }
    save_metrics_report(
        metrics_report=metrics_report,
        output_dir=output_dir_path,
        config=config,
    )
    for metric in ["loss", "accuracy", "precision", "recall", "f1_score"]:
        if metric in history.history:
            logging.info(f"Plotting metric curve for {metric}...")
            plot_metric_curve(
                history=history,
                metric_name=metric,
                output_dir_path=output_dir_path,
            )
            logging.info(f"{metric.capitalize()} curve saved.")

    logging.info("Experiment completed successfully.")


if __name__ == "__main__":
    logging.info("Parsing configuration from CLI...")
    config = CLI(BaldOrNotConfig)

    logging.info("Running experiment with parsed configuration.")
    run_experiment(config=config)
