#!/usr/bin/env/ python
# ECBM E4040 Fall 2017 Assignment 2
# Optimizer implementations

import numpy as np


class Optimizer(object):
    def train(self, model, X_train, y_train, X_valid, y_valid,
              num_epoch=10, batch_size=500, learning_rate=1e-3, learning_decay=0.95, verbose=False, record_interval=10):

        """
        This function is for training

        Inputs:
        :param model: (class MLP) a MLP model
        :param X_train: (float32) input data, a tensor with shape (N, D1, D2, ...)
        :param y_train: (int) label data for classification, a 1D array of length N
        :param X_valid: (float32) input data, a tensor with shape (num_valid, D1, D2, ...)
        :param y_valid: (int) label data for classification, a 1D array of length num_valid
        :param num_epoch: (int) the number of training epochs
        :param batch_size: (int) the size of a single batch for training
        :param learning_rate: (float)
        :param learning_decay: (float) reduce learning rate every epoch
        :param verbose: (boolean) whether report training process
        """
        num_train = X_train.shape[0]
        num_batch = num_train // batch_size
        print('number of batches for training: {}'.format(num_batch))
        loss_hist = []
        train_acc_hist = []
        valid_acc_hist = []
        loss = 0.0
        for e in range(num_epoch):
            # Train stage
            for i in range(num_batch):
                # Order selection
                X_batch = X_train[i * batch_size:(i + 1) * batch_size]
                y_batch = y_train[i * batch_size:(i + 1) * batch_size]
                # loss
                loss += model.loss(X_batch, y_batch)
                # update model
                self.step(model, learning_rate=learning_rate)

                if (i + 1) % record_interval == 0:
                    loss /= record_interval
                    loss_hist.append(loss)
                    if verbose:
                        print('{}/{} loss: {}'.format(batch_size * (i + 1), num_train, loss))
                    loss = 0.0

            # Validation stage
            train_acc = model.check_accuracy(X_train, y_train)
            val_acc = model.check_accuracy(X_valid, y_valid)
            train_acc_hist.append(train_acc)
            valid_acc_hist.append(val_acc)
            # Shrink learning_rate
            learning_rate *= learning_decay
            print('epoch {}: valid acc = {}, new learning rate = {}'.format(e + 1, val_acc, learning_rate))

        # Save loss and accuracy history
        self.loss_hist = loss_hist
        self.train_acc_hist = train_acc_hist
        self.valid_acc_hist = valid_acc_hist

        return loss_hist, train_acc_hist, valid_acc_hist

    def test(self, model, X_test, y_test, batch_size=10000):
        """
        Inputs:
        :param model: (class MLP) a MLP model
        :param X_test: (float) a tensor of shape (N, D1, D2, ...)
        :param y_test: (int) an array of length N
        :param batch_size: (int) seperate input data into several batches
        """
        acc = 0.0
        num_test = X_test.shape[0]

        if num_test <= batch_size:
            acc = model.check_accuracy(X_test, y_test)
            print('accuracy in a small test set: {}'.format(acc))
            return acc

        num_batch = num_test // batch_size
        for i in range(num_batch):
            X_batch = X_test[i * batch_size:(i + 1) * batch_size]
            y_batch = y_test[i * batch_size:(i + 1) * batch_size]
            acc += batch_size * model.check_accuracy(X_batch, y_batch)

        X_batch = X_test[num_batch * batch_size:]
        y_batch = y_test[num_batch * batch_size:]
        if X_batch.shape[0] > 0:
            acc += X_batch.shape[0] * model.check_accuracy(X_batch, y_batch)

        acc /= num_test
        print('test accuracy: {}'.format(acc))
        return acc

    def step(self, learning_rate):
        pass


class SGDOptim(Optimizer):
    def __init__(self):
        pass

    def step(self, model, learning_rate):
        """
        Implement a one-step SGD update on network's parameters
        
        Inputs:
        :param model: a neural network class object
        :param learning_rate: (float)
        """
        # get all parameters and their gradients
        params = model.params
        grads = model.grads

        for k in grads:
            ## update each parameter
            params[k] -= learning_rate * grads[k]


class SGDmomentumOptim(Optimizer):
    def __init__(self, model, momentum=0.5):
        """
        Inputs:
        :param model: a neural netowrk class object
        :param momentum: (float)
        """
        self.momentum = momentum
        velocitys = dict()
        for k, v in model.params.items():
            velocitys[k] = np.zeros_like(v)
        self.velocitys = velocitys

    def step(self, model, learning_rate):
        """
        Implement a one-step SGD+momentum update on network's parameters
        
        Inputs:
        :param model: a neural network class object
        :param learning_rate: (float)
        """
        momentum = self.momentum
        velocitys = self.velocitys# dictionary
        # get all parameters and their gradients
        params = model.params
        grads = model.grads
        ###################################################
        # TODO: SGD + Momentum, Update params and velocitys#
        # print("velocity keys: {}".format(velocitys.keys()))
        # print("param keys: {}".format(params.keys()))
        # print("grads keys: {}".format(grads.keys()))
        for paramKey in velocitys.keys():
            velocitys[paramKey] = momentum*velocitys[paramKey] - learning_rate*grads[paramKey]
            params[paramKey] = params[paramKey] + velocitys[paramKey]
            # estimate gradient 
            # update velocity
            # update parameters
        ###################################################
        # raise NotImplementedError


class RMSpropOptim(Optimizer):
    def __init__(self, model, gamma=0.9, eps=1e-12):
        """
        Inputs:
        :param model: a neural network class object
        :param gamma: (float) suggest to be 0.9
        :param eps: (float) a small number
        """
        # eps = 1e-1
        self.gamma = gamma
        self.eps = eps
        cache = dict()
        for k, v in model.params.items():
            cache[k] = np.zeros_like(v)
        self.cache = cache

    def step(self, model, learning_rate):
        """
        Implement a one-step RMSprop update on network's parameters
        And a good default learning rate can be 0.001.
        
        Inputs:
        :param model: a neural network class object
        :param learning_rate: (float)
        """
        gamma = self.gamma
        eps = self.eps
        cache = self.cache
        # decayRate = learning_rate # why is this named differently?

        # I'm renaming all of these because the names are different compared to
        # what is used in the book. Renaming them according to what the general consensus on values is
        decayRate = gamma# should be arounf 0.9.
        gamma = eps# should be a very small number like 1e-6
        eps = learning_rate

        # create two new dictionaries containing all parameters and their gradients
        params, grads = model.params, model.grads
        # print("cache keys: {}".format(cache.keys()))
        # print("param keys: {}".format(params.keys()))
        # print("grads keys: {}".format(grads.keys()))
        ###################################################
        # TODO: RMSprop, Update params and cache           #
        for paramKey in cache.keys():
            # print("{} mean is {}".format(paramKey, params[paramKey].mean()))
            sqrdGrad = decayRate*cache[paramKey] + (1-decayRate)*np.square(grads[paramKey])
            deltaGrad = -(eps/np.sqrt(gamma + sqrdGrad))*grads[paramKey]
            model.params[paramKey] = params[paramKey] + deltaGrad
            self.cache[paramKey] = sqrdGrad
        ###################################################
        # raise NotImplementedError


class AdamOptim(Optimizer):
    def __init__(self, model, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Inputs:
        :param model: a neural network class object
        :param beta1: (float) should be close to 1
        :param beta2: (float) similar to beta1
        :param eps: (float) in different case, the good value for eps will be different
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        momentums = dict()
        velocitys = dict()
        for k, v in model.params.items():
            momentums[k] = np.zeros_like(v)
            velocitys[k] = np.zeros_like(v)
        self.momentums = momentums
        self.velocitys = velocitys
        self.t = 0

    def step(self, model, learning_rate):
        """
        Implement a one-step Adam update on network's parameters
        
        Inputs:
        :param model: a neural network class object
        :param learning_rate: (float)
        """
        beta1 = self.beta1
        beta2 = self.beta2
        eps = self.eps

        momentums = self.momentums
        velocitys = self.velocitys
        self.t = self.t + 1 # I added this
        t = self.t
        # create two new dictionaries containing all parameters and their gradients
        params, grads = model.params, model.grads
        ###################################################
        # TODO: Adam, Update t, momentums, velocitys and   #
        # params                                           #
        ###################################################
        # print("Momentum keys {}".format(momentums.keys()))
        # print("velocitys keys {}".format(velocitys.keys()))
        for paramKey in params.keys():
            s = beta1*momentums[paramKey] + (1-beta1)*grads[paramKey]
            r = beta2*velocitys[paramKey] + (1-beta2)*np.square(grads[paramKey])
            s_hat = s/(1-beta1**t)
            r_hat = r/(1-beta2**t)
            deltaGrad = -learning_rate*(s_hat/(np.sqrt(r_hat) + eps))# seesm like names got switched around again
            params[paramKey] = params[paramKey] + deltaGrad
            momentums[paramKey] = s
            velocitys[paramKey] = r
        # raise NotImplementedError