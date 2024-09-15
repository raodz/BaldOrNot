from dataclasses import asdict
from datetime import datetime

import tensorflow as tf
import os
import logging

from src.data import BaldDataset
from src.model import BaldOrNotModel
from src.config_class import BoldOrNotConfig
from src.utils import check_log_exists_decorator


@check_log_exists_decorator
def train_model(config: BoldOrNotConfig, output_dir_path: str):
    """
    Trains the BaldOrNot model using the specified configuration.

    This function initializes the dataset, constructs the model, compiles it
    with the specified optimizer, loss function, and metrics, and then trains
    the model on the dataset for the number of epochs defined in the
    configuration.

    Args:
        config (BoldOrNotConfig): The configuration object containing model,
        training, and path parameters.
    """

    logging.info("Starting model training...")

    vector_dim = config.model_params.dense_units
    batch_size = config.training_params.batch_size

    # Log model parameters
    logging.info(
        f"Model configuration: Dense units: {vector_dim}, "
        f"Batch size: {batch_size}"
    )

    train_dataset = BaldDataset(batch_size=batch_size, vector_dim=vector_dim)
    logging.info(
        f"Training dataset initialized with batch size {batch_size}"
        f"and vector dim {vector_dim}"
    )

    # Initialize model
    model = BaldOrNotModel(**asdict(config.model_params))
    logging.info("Model initialized")

    # Compile model
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.training_params.learning_rate
    )
    model.compile(
        optimizer=optimizer,
        loss=config.training_params.loss_function,
        metrics=config.metrics,
    )
    logging.info(
        f"Model compiled with Adam optimizer and learning rate "
        f"{config.training_params.learning_rate}"
    )

    # Initialize callbacks
    tf_callbacks = []
    for callback_dict in config.callbacks:
        if callback_dict["type"] == "EarlyStopping":
            tf_callbacks.append(
                tf.keras.callbacks.EarlyStopping(**callback_dict["args"])
            )
            logging.info(
                f"EarlyStopping callback added with parameters: "
                f"{callback_dict['args']}"
            )
        elif callback_dict["type"] == "TensorBoard":
            tf_callbacks.append(
                tf.keras.callbacks.TensorBoard(**callback_dict["args"])
            )
            logging.info(
                f"TensorBoard callback added with parameters: "
                f"{callback_dict['args']}"
            )

    # Train the model
    logging.info(
        f"Starting training for {config.training_params.epochs} epochs"
    )
    history = model.fit(
        train_dataset,
        epochs=config.training_params.epochs,
        validation_data=train_dataset,
        callbacks=tf_callbacks,
    )
    logging.info("Model training completed")

    # Save model and plot
    model_path = os.path.join(
        output_dir_path, config.model_params.saved_model_name
    )
    model.save(model_path)
    logging.info(f"Model saved at {model_path}")

    return history


def init_output_dir(training_name: str) -> str:
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_training = f"{training_name}{current_date}"
    output_dir_path = os.path.join(project_path, "trainings", current_training)
    os.makedirs(output_dir_path, exist_ok=True)
    return output_dir_path
