#!/usr/bin/env/ python
# ECBM E4040 Fall 2017 Assignment 2
# This Python script contains the ImageGenrator class.

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate


class ImageGenerator(object):

    def __init__(self, x, y):
        """
        Initialize an ImageGenerator instance.
        :param x: A Numpy array of input data. It has shape (num_of_samples, height, width, channels).
        :param y: A Numpy vector of labels. It has shape (num_of_samples, ).
        """

        # TODO: Your ImageGenerator instance has to store the following information:
        # x, y, num_of_samples, height, width, number of pixels translated, degree of rotation, is_horizontal_flip,
        # is_vertical_flip, is_add_noise. By default, set boolean values to
        self.x = np.copy(x)#np.array(x, dtype = 'float64')
        self.y = y
        self.N, self.height, self.width, _= x.shape
        self.nPixelsTranslated = None# not sure what default this should be
        self.rotDegree = None# not sure what default this should be
        self.is_horizontal_flip = False
        self.is_vertical_flip = False
        self.is_add_noise = False
        # False.
        # raise NotImplementedError
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################

    def next_batch_gen(self, batch_size, shuffle=True):
        """
        A python generator function that yields a batch of data indefinitely.
        :param batch_size: The number of samples to return for each batch.
        :param shuffle: If True, shuffle the entire dataset after every sample has been returned once.
                        If False, the order or data samples stays the same.
        :return: A batch of data with size (batch_size, width, height, channels).
        """

        # TODO: Use 'yield' keyword, implement this generator. Pay attention to the following:
        # 1. The generator should return batches endlessly.
        # 2. Make sure the shuffle only happens after each sample has been visited once. Otherwise some samples might
        # not be output.

        # One possible pseudo code for your reference:
        #######################################################################
        #   calculate the total number of batches possible (if the rest is not sufficient to make up a batch, ignore)
        #   while True:
        #       if (batch_count < total number of batches possible):
        #           batch_count = batch_count + 1
        #           yield(next batch of x and y indicated by batch_count)
        #       else:
        #           shuffle(x)
        #           reset batch_count
        totalBatches = self.N//batch_size
        #print("total batches {}".format(totalBatches))
        batchCount = 0
        randIndices = np.random.randint(low = 0, high = self.N, size = self.N)
        randIndices = np.random.permutation(self.N)
        while True:
            if batchCount < totalBatches:
                # batchCount += 1
                myIndices = randIndices[batch_size*batchCount:(batch_size*(batchCount + 1))]
                yield(self.x[myIndices,:], self.y[myIndices])
                batchCount += 1
            else:
                if shuffle:
                    #print("Shuffling X\nX shape before: {}".format(self.x.shape))
                    myShuffle = np.random.permutation(self.N)
                    self.x = self.x[myShuffle, :]# shuffle x
                    self.y = self.y[myShuffle]
                batchCount = 0
                #print("X shape after: {}".format(self.x.shape))
        # raise NotImplementedError
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################

    def show(self):
        """
        Plot the top 16 images (index 0~15) of self.x for visualization.
        """
        
      
        
        
        
        
        r = 4
        f, axarr = plt.subplots(r, r, figsize=(8,8))
        counter = 0
        for i in range(r):
            for j in range(r):
                temp = self.x[counter,:]
                counter += 1
                img = self.x[counter,:]
                axarr[i][j].imshow(img)
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################

    def translate(self, shift_height, shift_width):
        """
        Translate self.x by the values given in shift.
        :param shift_height: the number of pixels to shift along height direction. Can be negative.
        :param shift_width: the number of pixels to shift along width direction. Can be negative.
        :return:
        """

        # TODO: Implement the translate function. Remember to record the value of the number of pixels translated.
        # Note: You may wonder what values to append to the edge after the translation. Here, use rolling instead. For
        # example, if you translate 3 pixels to the left, append the left-most 3 columns that are out of boundary to the
        # right edge of the picture.
        # Hint: Numpy.roll
        # (https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.roll.html)
        self.x = np.roll(self.x, shift_height, axis=0)# rolls along the rows
        self.x = np.roll(self.x, shift_width, axis=1)# rolls along the columns
        self.nPixelsTranslated = (shift_height*self.height + shift_width*self.width)*self.N

        # raise NotImplementedError
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################

    def rotate(self, angle=0.0):
        """
        Rotate self.x by the angles (in degree) given.
        :param angle: Rotation angle in degrees.

        - https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.interpolation.rotate.html
        """
        # TODO: Implement the rotate function. Remember to record the value of
        # rotation degree.
        self.rotDegree = angle
        self.x = rotate(self.x, angle = angle, axes=(0, 1), reshape=False, 
                     output=None, order=3, mode='constant', cval=0.0, prefilter=True)
        # This rotation isn't working correctly. Get shit for non right anlge rotatations
        # raise NotImplementedError
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################

    def flip(self, mode='h'):
        """
        Flip self.x according to the mode specified
        :param mode: 'h' or 'v' or 'hv'. 'h' means horizontal and 'v' means vertical.
        """
        # TODO: Implement the flip function. Remember to record the boolean values is_horizontal_flip and
        # is_vertical_flip.
        if mode == 'h':
            self.is_horizontal_flip = True
            self.x = np.flipud(self.x)
        elif mode == 'v':
            self.is_vertical_flip = True
            self.x = np.fliplr(self.x)
        else:
            self.is_vertical_flip = True
            self.is_horizontal_flip = True
            self.x = np.fliplr(self.x)
            self.x = np.flipud(self.x)
        # raise NotImplementedError
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################

    def add_noise(self, portion, amplitude):
        """
        Add random integer noise to self.x.
        :param portion: The portion of self.x samples to inject noise. If x contains 10000 sample and portion = 0.1,
                        then 1000 samples will be noise-injected.
        :param amplitude: An integer scaling factor of the noise.
        """
        # TODO: Implement the add_noise function. Remember to record the
        # boolean value is_add_noise. You can try uniform noise or Gaussian
        # noise or others ones that you think appropriate.
        self.is_add_noise = True
        noise = np.random.normal(0, 1, size = self.x.shape)
        noNoiseIndices = np.random.choice(self.x.shape[0], size = int(np.round(self.x.shape[0]*(1 - portion))), replace = False)
        noise = (noise*amplitude).astype(np.uint8)
        noise[noNoiseIndices,:] = 0
        self.x += noise
        self.x = np.clip(self.x, 0, 255, out=self.x)
        
        
        
        
        # raise NotImplementedError
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
