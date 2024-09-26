import logging
import os
from dataclasses import asdict

import keras_tuner as kt
import pandas as pd
from datetime import datetime
import tensorflow as tf

from src.config_class import BaldOrNotConfig
from src.data import BaldDataset
from src.model import BaldOrNotModel
from src.utils import check_log_exists_decorator


@check_log_exists_decorator
def train_model(
    config: BaldOrNotConfig,
    output_dir_path: str,
    tune_hyperparams: bool = False,
):
    """
    Trains the BaldOrNot model using the specified configuration, with optional
    hyperparameter tuning using Keras Tuner.

    Args:
        config (BaldOrNotConfig): The configuration object containing model,
        training, and path parameters.
        output_dir_path (str): Directory where the trained model will be saved.
        tune_hyperparams (bool): Whether to perform hyperparameter tuning using
        Keras Tuner. Default is True.

    Returns:
        history: Training history object.
    """

    logging.info("Starting model training...")

    # Load datasets
    train_csv_path = os.path.join("..", "src", "data", "train.csv")
    train_df = pd.read_csv(train_csv_path)
    train_dataset = BaldDataset(
        train_df, batch_size=config.training_params.batch_size
    )
    logging.info(
        f"Training dataset initialized with batch size "
        f"{config.training_params.batch_size}"
    )

    val_csv_path = os.path.join("..", "src", "data", "val.csv")
    val_df = pd.read_csv(val_csv_path)
    val_dataset = BaldDataset(
        val_df, batch_size=config.training_params.batch_size
    )
    logging.info(
        f"Validation dataset initialized with batch size "
        f"{config.training_params.batch_size}"
    )

    def build_model(hp):
        """
        Build model using hyperparameters from Keras Tuner.
        """
        dense_units = hp.Choice(
            "dense_units",
            values=[32, 64, 128, 256, 512, 1024],
            default=config.model_params.dense_units,
        )
        dropout_rate = hp.Float(
            "dropout_rate",
            min_value=0.0,
            max_value=0.7,
            step=0.1,
            default=config.model_params.dropout_rate,
        )
        freeze_backbone = hp.Boolean(
            "freeze_backbone", default=config.model_params.freeze_backbone
        )

        model = BaldOrNotModel(
            dense_units=dense_units,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.training_params.learning_rate
            ),
            loss=config.training_params.loss_function,
            metrics=config.metrics,
        )

        return model

    if tune_hyperparams:
        logging.info("Starting hyperparameter tuning with Keras Tuner...")
        tuner = kt.RandomSearch(
            build_model,
            objective="val_accuracy",
            max_trials=20,  # Maximum number of trials for the tuner
            executions_per_trial=1,
            directory="tuner_logs",
            project_name="bald_or_not_tuning",
        )

        tuner.search(
            train_dataset,
            epochs=config.training_params.epochs,
            validation_data=val_dataset,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5
                )
            ],
        )

        # Get the best model from tuning
        best_model = tuner.get_best_models(num_models=1)[0]
        logging.info("Best hyperparameters found and model built.")
    else:
        # If not tuning, use the provided config to build the model directly
        logging.info("Building model with predefined hyperparameters...")
        best_model = BaldOrNotModel(**asdict(config.model_params))
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.training_params.learning_rate
        )
        best_model.compile(
            optimizer=optimizer,
            loss=config.training_params.loss_function,
            metrics=config.metrics,
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

    # Train the best model
    logging.info(
        f"Starting training for {config.training_params.epochs} epochs"
    )
    history = best_model.fit(
        train_dataset,
        epochs=config.training_params.epochs,
        validation_data=val_dataset,
        callbacks=tf_callbacks,
    )
    logging.info("Model training completed")

    # Save the best model
    model_path = os.path.join(
        output_dir_path, config.model_params.saved_model_name
    )
    best_model.save(model_path)
    logging.info(f"Model saved at {model_path}")

    return history


def init_output_dir(training_name: str) -> str:
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_training = f"{training_name}{current_date}"
    output_dir_path = os.path.join(project_path, "trainings", current_training)
    os.makedirs(output_dir_path, exist_ok=True)
    return output_dir_path
