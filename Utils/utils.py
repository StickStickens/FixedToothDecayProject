import os
def find_project_root(project_name="FixedToothDecayProject"):
    """Find the project root directory by walking up parent folders.

    Parameters
    ----------
    project_name : str, default="FixedToothDecayProject"
        Name of the root folder to locate.

    Returns
    -------
    str
        Absolute path to the project root.

    Raises
    ------
    FileNotFoundError
        If the root folder cannot be found in any parent directory.
    """
    current_path = os.path.abspath(os.path.dirname(__file__))

    while True:
        if os.path.basename(current_path) == project_name:
            return current_path

        parent = os.path.dirname(current_path)
        if parent == current_path:
            raise FileNotFoundError(f"Could not find project root '{project_name}'")

        current_path = parent

from sklearn.metrics import roc_auc_score
import numpy as np
def evaluate_model(y_test, y_pred, y_proba):
    """Compute weighted OvO ROC-AUC for model predictions.

    Parameters
    ----------
    y_test : numpy.ndarray
        Ground-truth encoded class labels.
    y_pred : numpy.ndarray
        Predicted encoded class labels. Included for API consistency.
    y_proba : numpy.ndarray
        Predicted class probabilities with shape `(n_samples, n_classes)`.

    Returns
    -------
    float | str
        Weighted one-vs-one ROC-AUC score. Returns "-----" when the metric
        cannot be computed because only one class is present.
    """
    n_classes = y_proba.shape[1]
    y_bin = np.stack([(y_test == i).astype(np.int32) for i in range(n_classes)], axis=1)
    if len(np.unique(y_bin)) <= 1:
        return "-----"
    return roc_auc_score(y_bin, y_proba, average="weighted", multi_class="ovo")
