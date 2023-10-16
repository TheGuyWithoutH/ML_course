import numpy as np
from costs import compute_loss 
from gradient_descent import compute_gradient
def reg_logistic_regression(y, tx, lambda_,initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""
    # init parameters
    #threshold = 1e-8
    gradient= compute_gradient(y, tx, initial_w, max_iters, gamma)
    """ losses = []
    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break """
    N = tx.shape[0]
    a = initial_w- (1/N)*np.sum(np.log(1+np.exp(tx@initial_w))-y@tx@initial_w)+lambda_*np.linalg.norm(initial_w)**2
    raise NotImplementedError