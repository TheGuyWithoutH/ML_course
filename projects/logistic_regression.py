from costs import *
from gradient_descent import *


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent or SGD """
    losses = []
    ws = [initial_w]
    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss = compute_loss(y, tx, w)
        grad = compute_gradient(y, tx, w, max_iters, gamma)
        w = w - gamma*grad
        losses.append(loss)
        ws.append(w)

    return ws, losses
