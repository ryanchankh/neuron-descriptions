import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve


def compute_correct(y_pred, y_true):
    """Count number of correct classification. """
    assert y_pred.shape == y_true.shape
    return (y_pred == y_true).sum()


def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    return 1 - compute_correct(y_pred, y_true) / y_true.size(0)

def compute_sensitivity_specificity(y_pred, y_true):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

def compute_precision_recall(y_pred_prob, y_true):
    lst_precision, lst_recall, lst_thresholds = precision_recall_curve(y_true, y_pred_prob[:, 1])
    return lst_precision, lst_recall, lst_thresholds

def compute_balanced_accuracy(y_pred, y_true):
    sensitivity, specificity = compute_sensitivity_specificity(y_pred, y_true)
    return 0.5 * (sensitivity + specificity)

