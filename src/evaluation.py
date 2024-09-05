import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def make_predictions(model, dataset):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        predictions = model.predict(images)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels)

    y_true = np.array(y_pred)
    y_pred = np.array(y_pred)

    return y_true, y_pred


def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, precision, recall, f1


def misclassifications(y_true, y_pred):
    # First type
    false_positives = np.where((y_pred == 1) & (y_true == 0))[0]
    # Second type
    false_negatives = np.where((y_pred == 0) & (y_true == 1))[0]

    return false_positives, false_negatives

