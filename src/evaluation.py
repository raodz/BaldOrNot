import numpy as np


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




