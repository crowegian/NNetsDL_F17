import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wst W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)


    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    nClasses = W.shape[0]

    loss = 0.0
    dW = np.zeros_like(W)

    n = X.shape[0]
    for i in range(0, n): # iterating through each observation
        scores = np.dot(X[i,:], W)
        scores = np.exp(scores - np.max(scores))

        expDenom = 0.0
        for score in scores:
            expDenom = expDenom + score
        loss = loss - np.log(scores[y[i]]/expDenom)

        for j in range(0, W.shape[1]):
            if j == y[i]:
                dW[:,j] += -X[i,:]+(scores[j]/expDenom)*X[i,:]
            else:
                dW[:,j] += (scores[j]/expDenom)*X[i,:]

    loss = loss/n # averaging over all examples
    dW = dW/n

    # Add regularization to the loss.
    loss = loss + 0.5*reg*np.sum(W*W)

    # Add regularization to the gradient
    dW = dw + reg*W

    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    #############################################################################

    n = X.shape[0]
    scores = np.dot(X, W)
    scores = (scores.T - np.max(scores, axis = 1)).T
    # 1/0
    scores = np.exp(scores)
    scoresDenom = np.sum(scores, axis = 1) # sum of scores along observations
    loss = -np.log(((scores.T)/scoresDenom).T)
    loss = loss[np.arange(0,n), y]
    loss = np.sum(loss)/n + 0.5*reg*np.sum(W*W)
    # print(loss)





    dW = ((scores.T)/scoresDenom).T
    # print(dW.shape)
    # 1/0
    dW[np.arange(0,n), y] = dW[np.arange(0,n), y] - 1
    # print('Dw shape {}'.format(dW.shape))
    # print('X shape {}'.format(X.shape))

    dW = np.dot(X.T, dW)/n + reg*W
    # dW = dW/n + reg*W

    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
