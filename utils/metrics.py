import numpy as np

def compute_confusion_matrix(y_true, y_pred, num_classes=2):
    """
    Computes a confusion matrix for binary or multi-class classification.

    Args:
        y_true (list or np.array): True labels
        y_pred (list or np.array): Predicted labels
        num_classes (int): Number of classes (default 2 for binary)

    Returns:
        np.array: Confusion matrix of shape (num_classes, num_classes)
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm
