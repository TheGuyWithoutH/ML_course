import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        loss = compute_loss(y, tx, w, 'mse')
        gradient = compute_gradient(y, tx, w, 'mse')
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        # ***************************************************
        w = w - gamma * gradient
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    w = initial_w
    batches = batch_iter(y, tx, batch_size, num_batches=max_iters)

    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: implement stochastic gradient descent.
        # ***************************************************
        loss = compute_loss(y, tx, w, 'mse')
        batch_y, batch_tx = next(batches)
        gradient = compute_gradient(batch_y, batch_tx, w, 'mse')
        w = w - gamma * gradient

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    w = np.linalg.solve(tx.T@tx, tx.T@y)
    return w, compute_loss(y, tx, w, 'mse')


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
    w = np.linalg.solve(A, b)
    # return w and loss function

    return w, compute_loss(y, tx, w, 'mse')


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent or SGD

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape (D,), initial weights.
        max_iters: scalar, number of iterations.
        gamma: scalar, step size.
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar, loss function value.
    """

    w = initial_w
    # start the logistic regression
    for n_iter in range(max_iters):
        # get loss and update w.
        loss = compute_loss(y, tx, w, 'log')
        grad = compute_gradient(y, tx, w, 'log')
        w = w - gamma*grad

        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression using gradient descent or SGD

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, regularization strength.
        initial_w: numpy array of shape (D,), initial weights.
        max_iters: scalar, number of iterations.
        gamma: scalar, step size.
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar, loss function value.
    """
    w = initial_w
    # start the logistic regression
    for n_iter in range(max_iters):
        # get loss and update w.
        loss = compute_loss(y, tx, w, 'log')
        grad = compute_gradient(y, tx, w, 'log') + 2*lambda_*w
        w = w - gamma*grad

        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return w, loss


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def calculate_logloss(y, y_pred, eps=1e-8):
    """Calculate the logloss for vector y and y_pred."""
    return -np.mean(y*np.log(y_pred + eps) + (1-y)*np.log(1-y_pred + eps))


def sigmoid(t):
    """apply sigmoid function on t."""
    return 1/(1 + np.exp(-t))


def compute_loss(y, tx, w, loss_type):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.
        loss_type: string in ["mae", "mse", "log"] specifying the type of loss to compute

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    e = y - tx @ w

    if loss_type == "mse":
        return calculate_mse(e)

    elif loss_type == "mae":
        return calculate_mae(e)

    elif loss_type == "log":
        y_pred = sigmoid(tx @ w)
        return calculate_logloss(y, y_pred)
    else:
        raise ValueError("Unknown loss function: " + loss_type)


def compute_gradient(y, tx, w, loss_type):
    """Compute the gradient.
    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.
        loss_type: string in ["mse", "log"] specifying the type of loss to compute
    Returns:
        grad: shape=(2,). The gradient of the loss with respect to w.
    """
    if loss_type == "mse":
        return -tx.T @ (y - tx @ w) / len(y)
    elif loss_type == "log":
        return tx.T @ (sigmoid(tx @ w) - y) / len(y)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
