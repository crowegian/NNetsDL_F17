from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine function.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: a numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: a numpy array of weights, of shape (D, M)
    - b: a numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    """
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    D = np.prod(x.shape[1:])# multiply all dimensions from the second dimension 
    # onwards to get D
    x = np.reshape(x, (x.shape[0], D))
    out = np.dot(x, w) + b

    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out


def affine_backward(dout, x, w, b):
    """
    Computes the backward pass of an affine function.

    Inputs:
    - dout: upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: input data, of shape (N, d_1, ... d_k)
      - w: weights, of shape (D, M)
      - b: bias, of shape (M,)

    Returns a tuple of:
    - dx: gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: gradient with respect to w, of shape (D, M)
    - db: gradient with respect to b, of shape (M,)
    """
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################


    D = np.prod(x.shape[1:])# multiply all dimensions from the second dimension 
    # onwards to get D
    x = np.reshape(x, (x.shape[0], D))

    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(x.T, dout)
    db = np.sum(dout, axis=0)

    # print('dx shape {}'.format(dx.shape))
    # print('dw shape {}'.format(dw.shape))
    # print('db shape {}'.format(db.shape))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for rectified linear units (ReLUs).

    Input:
    - x: inputs, of any shape

    Returns a tuple of:
    - out: output, of the same shape as x
    """
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # mask = x < 0
    out = np.array(x, copy=True)
    # consider copying here if you get unexplained issues
    out[x <= 0] = 0

    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out


def relu_backward(dout, x):
    """
    Computes the backward pass for rectified linear units (ReLUs).

    Input:
    - dout: upstream derivatives, of any shape

    Returns:
    - dx: gradient with respect to x
    """
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################

    # D = np.prod(x.shape[1:])# multiply all dimensions from the second dimension 
    # onwards to get D
    # x = np.reshape(x, (x.shape[0], D))
    dx = np.array(dout, copy=True)
    # consider copying here if you get unexplained issues
    # print(x[0] < 0)
    dx[x <= 0] = 0 # the <= is key because there are zero values which you want
    # to catch
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    y_prediction = argmax(softmax(x))
    
    Inputs:
    - X: (float) a tensor of shape (N, #classes)
    - y: (int) ground truth label, a array of length N

    Returns:
    - loss: the cross-entropy loss
    - dx: gradients wrt input x
    """
    # Initialize the loss.
    loss = 0.0
    dx = np.zeros_like(x)
    #############################################################################
    # TODO: You can use the previous softmax loss function here.                #
    #############################################################################

    # sm = np.exp(x - np.max(x, keepdims = True, axis = 1))#np.reshape(np.max(x, axis=1), (x.shape[0], 1)))# stability should be fine here
    sm = x - np.max(x, keepdims = True, axis = 1)

    N = x.shape[0]



    sm = -sm + np.log(np.sum(np.exp(sm), axis=1, keepdims=True))

    N = x.shape[0]


    loss = np.sum(sm[np.arange(N), y])/N




    dx =  x - np.max(x, keepdims = True, axis = 1)
    dx = np.exp(dx)
    scoresDenom = np.sum(dx, axis = 1, keepdims=True) # sum of scores along observations

    dx = ((dx)/scoresDenom)

    dx[np.arange(0,N), y] = dx[np.arange(0,N), y] - 1
    dx = dx/N





    # dx = x - np.max(x, keepdims = True, axis = 1)
    # # dx = 
    # dx = sm
    # dx[np.arange(N), y] = dx[np.arange(N), y] - 1
    # dx = dx/N
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dx