from builtins import range
from builtins import object
import numpy as np

from ecbm4040.layer_funcs import *
from ecbm4040.layer_utils import *

class MLP(object):
    """
    MLP with an arbitrary number of dense hidden layers,
    and a softmax loss function. For a network with L layers,
    the architecture will be

    input >> DenseLayer x (L - 1) >> AffineLayer >> softmax_loss >> output

    Here "x (L - 1)" indicate to repeat L - 1 times. 
    """
    def __init__(self, input_dim=3072, hidden_dims=[200,200], num_classes=10, reg=0.0, weight_scale=1e-2):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """
        self.num_layers = len(hidden_dims) + 1
        self.reg = reg
        
        dims = [input_dim] + hidden_dims
        layers = []
        # print('all Dims {}'.format(dims))
        for i in range(len(dims)-1):
            # print('curr dim: {}\n next dim {}'.format(dims[i], dims[i+1]))
            layers.append(DenseLayer(input_dim=dims[i], output_dim=dims[i+1], weight_scale=weight_scale))
        layers.append(AffineLayer(input_dim=dims[-1], output_dim=num_classes, weight_scale=weight_scale))
        # 1/0
        self.layers = layers

    def loss(self, X, y):
        """
        Calculate the cross-entropy loss and then use backpropogation
        to get gradients wst W,b in each layer.
        
        Inputs:
        - X: input data
        - y: ground truth
        
        Return loss value(float)
        """
        loss = 0.0
        reg = self.reg
        num_layers = self.num_layers
        layers = self.layers
        ###################################################
        #TODO: Feedforward                                #
        ###################################################
        hPrev = X# this will hold intermediate values
        hNext = X
        h = X
        for layer in layers:
            # print(h.shape)
            h = layer.feedforward(h)# update h to new input for next layer
            # hNext = layer.feedforward(hPrev)# update h to new input for next layer
            # hNext = hPrev
            # print('layer shape {}'.format(layer.params[0].shape))
        loss, dX = softmax_loss(h, y)
        
        ###################################################
        #TODO: Backpropogation                            #
        ###################################################
        # dX = lossGrad
        # print('dX shape {}'.format(dX.shape))
        for layer in reversed(layers):
            # print('hi')
            dX = layer.backward(dX)
        
        
        ###################################################
        # TODO: Add L2 regularization                     #
        ###################################################
        square_weights = 0
        for layer in layers:
            square_weights = np.sum(layer.params[0]**2)
        loss += 0.5*reg*square_weights
        
        
        ###################################################
        #              END OF YOUR CODE                   #
        ###################################################
        
        return loss

    def step(self, learning_rate=1e-5):
        """
        Use SGD to implement a single-step update to each weight and bias.
        """
        ###################################################
        #TODO: Use SGD or SGD with momentum to update     #
        #variables in layers                              #
        ###################################################
        # layer1, layer2 = self.layer1, self.layer2
        params = []#layer1.params + layer2.params
        grads = []#layer1.gradients + layer2.gradients
        layers = self.layers
        for layer in layers:
            params = params + layer.params
            grads = grads + layer.gradients        
        # Add L2 regularization
        reg = self.reg
        grads = [grad + reg*params[i] for i, grad in enumerate(grads)]
        for idx in range(0, len(params)):
            params[idx] = params[idx] - grads[idx]*learning_rate        
        ###################################################
        #              END OF YOUR CODE                   #
        ###################################################
   
        # update parameters in layers
        num_layers = len(layers)
        for i in range(num_layers):
            self.layers[i].update_layer(params[2*i:2*(i+1)])
        

    def predict(self, X):
        """
        Return the label prediction of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        
        Returns: 
        - predictions: (int) an array of length N
        """
        predictions = None
        num_layers = self.num_layers
        layers = self.layers
        #######################################################
        #TODO: Remember to use functions in class SoftmaxLayer#
        #######################################################
        h = X# this will hold intermediate values
        for layer in layers:
            h = layer.feedforward(h)# update h to new input for next layer
        predictions = (h.T - np.max(h, axis = 1)).T
        predictions = np.exp(predictions)
        denom = np.sum(predictions, axis = 1) # sum of scores along observations
        # loss = -np.log(((scores.T)/scoresDenom).T)
        predictions = (predictions.T/denom).T
        predictions = predictions.argmax(axis = 1)
        
        #######################################################
        #                 END OF YOUR CODE                    #
        #######################################################
        
        return predictions
    
    def check_accuracy(self, X, y):
        """
        Return the classification accuracy of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        - y: (int) an array of length N. ground truth label 
        Returns: 
        - acc: (float) between 0 and 1
        """
        y_pred = self.predict(X)
        acc = np.mean(np.equal(y, y_pred))
        
        return acc
        
        


