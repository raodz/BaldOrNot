import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


def make_predictions(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Makes predictions on the provided dataset and returns the true and
    predicted labels.

    Args:
        model: The trained machine learning model used for making predictions.
        dataset: The dataset to make predictions on. This should be an iterable
            of (images, labels).

    Returns:
        A tuple containing two numpy arrays:
        - y_true: The true labels.
        - y_pred: The predicted labels.
    """
    y_true = []
    y_pred = []

    for images, labels in dataset:
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return y_true, y_pred


def evaluate_model(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str: float, str: float, str:float, str:float]:
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

    return {'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1}


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


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: str,
) -> None:
    """
    Plots the confusion matrix and saves it as an image file.

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.
        class_names: List of class names to label the axes.
        output_path: Path where the confusion matrix image will be saved.

    Returns:
        None
    """
    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        matrix,
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
    print(f"Confusion matrix saved to {output_path}")
