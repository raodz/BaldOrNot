import numpy as np
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from src.constants import BALD_LABELS
from src.utils import check_log_exists_decorator


def make_predictions(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Makes y_pred on the provided dataset and returns the true and
    predicted labels.

    Args:
        model: The trained machine learning model used for making y_pred.
        dataset: The dataset to make y_pred on. This should be an iterable
            of (images, labels).

    Returns:
        A tuple containing two numpy arrays:
        - y_true: The true labels.
        - y_pred: The predicted labels.
    """
    images, y_true = zip(*dataset)
    images = np.concatenate(images, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    y_pred = model.predict(images)

    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy()

    return y_true, y_pred


def get_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str: float, str: float, str:float, str:float, str:np.ndarray]:
    """
    Evaluates the model performance by calculating various metrics.

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.

    Returns:
        A tuple containing:
        - accuracy: The accuracy score.
        - precision: The precision score (weighted).
        - recall: The recall score (weighted).
        - f1: The F1 score (weighted).
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    conf_matrix = confusion_matrix(y_true, y_pred)

    return {'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'conf_matrix': conf_matrix}


def get_misclassifications(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifies misclassifications by finding false positives
    and false negatives.

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.

    Returns:
        A tuple containing:
        - false_positives: Indices where false positives occurred.
        - false_negatives: Indices where false negatives occurred.
    """
    # First type: False Positives (predicted 1, but true label is 0)
    false_positives = np.where((y_pred == 1) & (y_true == 0))[0]

    # Second type: False Negatives (predicted 0, but true label is 1)
    false_negatives = np.where((y_pred == 0) & (y_true == 1))[0]

    return false_positives, false_negatives


def drop_confusion_matrix(
    confussion_matrix: np.ndarray,
    class_names: list[str],
    output_path: str,
) -> None:
    """
    Plots the confusion matrix and saves it as an image file.

    Args:
        class_names: List of class names to label the axes.
        output_path: Path where the confusion matrix image will be saved.

    Returns:
        None
    """

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        confussion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    plt.savefig(output_path)
    plt.close()
    logging.info(f"Confusion matrix saved to {output_path}")

@check_log_exists_decorator
def evaluate_and_save_results(model, dataset, dataset_name, output_dir_path):

    logging.info(f"Evaluating model on {dataset_name} set...")

    # Make predictions
    y_true, y_pred = make_predictions(model, dataset)

    # Calculate metrics
    metrics = get_metrics(y_true, y_pred)
    logging.info(f"{dataset_name.capitalize()} metrics: ")
    for metric, value in metrics.items():
        if metric != 'conf_matrix':
            logging.info(f"{metric}: {value:.4f}")

    # Save confusion matrix
    logging.info(f"Saving confusion matrix for {dataset_name} set...")
    conf_matrix_path = os.path.join(output_dir_path, f'{dataset_name}_confusion_matrix.png')
    drop_confusion_matrix(
        metrics['conf_matrix'],
        class_names=BALD_LABELS,
        output_path=conf_matrix_path
    )

    # Log misclassifications
    logging.info(f"Identifying misclassifications on {dataset_name} set...")
    false_positives, false_negatives = get_misclassifications(y_true, y_pred)
    logging.info(f"False positives: {len(false_positives)}, False negatives: {len(false_negatives)}")


