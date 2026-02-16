import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, classes=None, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plots the confusion matrix.

    Args:
        cm (np.array): Confusion matrix
        classes (list): List of class names
        title (str): Plot title
        cmap: Color map (default: plt.cm.Blues)
    """
    if classes is None:
        classes = [str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Add numbers on each cell
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
