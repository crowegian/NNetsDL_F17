import numpy as np

def svm_loss_naive(W, X, y, reg):
    """
    Multi-class Linear SVM loss function, naive implementation (with loops).
    
    In default, delta is 1 and there is no penalty term wst delta in objective function.

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: a numpy array of shape (D, C) containing weights.
    - X: a numpy array of shape (N, D) containing N samples.
    - y: a numpy array of shape (N,) containing training labels; y[i] = c means
         that X[i] has label c, where 0 <= c < C.
    - reg: (float) L2 regularization strength

    Returns:
    - loss: a float scalar
    - gradient: wrt weights W, an array of same shape as W
    """
    dW = np.zeros(W.shape).astype('float') # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += X[i]
                dW[:,y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg*2*W

    return loss, dW






def svm_loss_vectorized(W, X, y, reg):
    """
    Linear SVM loss function, vectorized implementation.
    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape).astype('float') # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################

    predictedClassScores = np.dot(X, W)
    correctClassScores = predictedClassScores[np.arange(0, predictedClassScores.shape[0]), y]
    correctClassScores = np.tile(correctClassScores.transpose(), (predictedClassScores.shape[1], 1)).transpose()
    loss = 1 + predictedClassScores - correctClassScores
    loss[loss < 0] = 0 # want margin to be greater than 0 if counting
    lossMat = loss
    loss[np.arange(0, predictedClassScores.shape[0]), y] = 0 # not counting the y_i class
    loss = np.sum(loss)
    loss = loss/X.shape[0]

    loss += reg*np.sum(W*W)

        
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################


    # loss += reg * np.sum(W * W)


    lossMat[lossMat > 0] = 1


    lossMat[np.arange(0, predictedClassScores.shape[0]), y] = -1 * np.sum(lossMat, axis=1)
    dW = np.dot(X.transpose(), lossMat)/X.shape[0]
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
    
