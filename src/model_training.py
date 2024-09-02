import tensorflow as tf
from src.data import BaldDataset
from src.model import BaldOrNotModel
from src.config import BoldOrNotConfig


def train_model(config: BoldOrNotConfig):
    """
    Trains the BaldOrNot model using the specified configuration.

    This function initializes the dataset, constructs the model, compiles it with the specified
    optimizer, loss function, and metrics, and then trains the model on the dataset for the
    number of epochs defined in the configuration.

    Args:
        config (BoldOrNotConfig): The configuration object containing model, training, and path parameters.
    """
    vector_dim = config.model_params.dense_units
    batch_size = config.training_params.batch_size

    train_dataset = BaldDataset(batch_size=batch_size, vector_dim=vector_dim)

    model = BaldOrNotModel(
                dense_units=config.model_params.dense_units,
                freeze_backbone=config.model_params.freeze_backbone,
                dropout_rate=config.model_params.dropout_rate
            )
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.training_params.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=config.training_params.loss_function,
        metrics=config.metrics
    )
    tf_callbacks = []
    for callback_dict in config.callbacks:
        if callback_dict['type'] == "EarlyStopping":
            tf_callbacks.append(tf.keras.callbacks.EarlyStopping(**callback_dict['args']))

    history = model.fit(
        train_dataset,
        epochs=config.training_params.epochs,
        validation_data=train_dataset,
        callbacks=tf_callbacks
    )

    return history


