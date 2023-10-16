import numpy as np

from costs import compute_loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    D = tx.shape[1]
    N = tx.shape[0]
    A = tx.T@tx + lambda_*2*N*np.identity(D)
    b = tx.T@y
    # return w and loss function

    return np.linalg.solve(A, b), compute_loss(y, tx, np.linalg.solve(A, b))
