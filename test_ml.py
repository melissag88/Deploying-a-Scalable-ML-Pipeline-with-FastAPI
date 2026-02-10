import pytest
# TODO: add necessary 
import numpy as np
from sklearn.linear_model import LogisticRegression
from ml.model import train_model, inference, compute_model_metrics

# TODO: implement the first test. Change the function name and input as needed
def test_train_model():
    """
    Test that train_model returns a LogisticRegression instance.
    """
    # Your code here
    X_dummy = np.array([[0, 1], [1, 0]])
    y_dummy = np.array([0, 1])
    model = train_model(X_dummy, y_dummy)
    assert isinstance(model, LogisticRegression)


# TODO: implement the second test. Change the function name and input as needed
def test_inference():
    """
    Test that inference returns a numpy array of predictions with the correct shape.
    """
    # Your code here
    X_dummy = np.array([[0, 1], [1, 0]])
    y_dummy = np.array([0, 1])
    model = train_model(X_dummy, y_dummy)
    preds = inference(model, X_dummy)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == y_dummy.shape


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns three floats: precision, recall, f1-score.
    """
    # Your code here
    y_true = np.array([0, 1, 1])
    y_preds = np.array([0, 1, 0])
    p, r, f1 = compute_model_metrics(y_true, y_preds)
    assert isinstance(p, float)
    assert isinstance(r, float)
    assert isinstance(f1, float)

