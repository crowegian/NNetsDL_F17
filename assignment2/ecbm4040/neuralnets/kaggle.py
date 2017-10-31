#!/usr/bin/env python
# ECBM E4040 Fall 2017 Assignment 2
# This script is intended for task 5 Kaggle competition. Use it however you want.
#!/usr/bin/env/ python
# ECBM E4040 Fall 2017 Assignment 2
# TensorFlow CNN

import tensorflow as tf
import numpy as np
import time
from ecbm4040.image_generator import ImageGenerator

####################################
# TODO: Build your own LeNet model #
####################################







class conv_layer(object):
    def __init__(self, input_x, in_channel, out_channel, kernel_shape, rand_seed, index=0):
        """
        :param input_x: The input of the conv layer. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
        :param in_channel: The 4-th demension (channel number) of input matrix. For example, in_channel=3 means the input contains 3 channels.
        :param out_channel: The 4-th demension (channel number) of output matrix. For example, out_channel=5 means the output contains 5 channels (feature maps).
        :param kernel_shape: the shape of the kernel. For example, kernal_shape = 3 means you have a 3*3 kernel.
        :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
        :param index: The index of the layer. It is used for naming only.
        """
        assert len(input_x.shape) == 4 and input_x.shape[1] == input_x.shape[2] and input_x.shape[3] == in_channel

        with tf.variable_scope('conv_layer_%d' % index):
            with tf.name_scope('conv_kernel'):
                w_shape = [kernel_shape, kernel_shape, in_channel, out_channel]
                weight = tf.get_variable(name='conv_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('conv_bias'):
                b_shape = [out_channel]
                bias = tf.get_variable(name='conv_bias_%d' % index, shape=b_shape,
                                       initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            # strides [1, x_movement, y_movement, 1]
            conv_out = tf.nn.conv2d(input_x, weight, strides=[1, 1, 1, 1], padding="SAME")
            cell_out = tf.nn.relu(conv_out + bias)

            self.cell_out = cell_out

            tf.summary.histogram('conv_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('conv_layer/{}/bias'.format(index), bias)

    def output(self):
        return self.cell_out


class max_pooling_layer(object):
    def __init__(self, input_x, k_size, padding="SAME"):
        """
        :param input_x: The input of the pooling layer.
        :param k_size: The kernel size you want to behave pooling action.
        :param padding: The padding setting. Read documents of tf.nn.max_pool for more information.
        """
        with tf.variable_scope('max_pooling'):
            # strides [1, k_size, k_size, 1]
            pooling_shape = [1, k_size, k_size, 1]
            cell_out = tf.nn.max_pool(input_x, strides=pooling_shape,
                                      ksize=pooling_shape, padding=padding)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class norm_layer(object):
    def __init__(self, input_x):
        """
        :param input_x: The input that needed for normalization.
        """
        with tf.variable_scope('batch_norm'):
            mean, variance = tf.nn.moments(input_x, axes=[0], keep_dims=True)
            cell_out = tf.nn.batch_normalization(input_x,
                                                 mean,
                                                 variance,
                                                 offset=None,
                                                 scale=None,
                                                 variance_epsilon=1e-6,
                                                 name=None)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class fc_layer(object):
    def __init__(self, input_x, in_size, out_size, rand_seed, keep_prob, activation_function=None, index=0):
        """
        :param input_x: The input of the FC layer. It should be a flatten vector.
        :param in_size: The length of input vector.
        :param out_size: The length of output vector.
        :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
        :param keep_prob: The probability of dropout. Default set by 1.0 (no drop-out applied)
        :param activation_function: The activation function for the output. Default set to None.
        :param index: The index of the layer. It is used for naming only.

        """
        with tf.variable_scope('fc_layer_%d' % index):
            #keep_prob = tf.placeholder(tf.float32)
            with tf.name_scope('fc_kernel'):
                w_shape = [in_size, out_size]
                weight = tf.get_variable(name='fc_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('fc_kernel'):
                b_shape = [out_size]
                bias = tf.get_variable(name='fc_bias_%d' % index, shape=b_shape,
                                       initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            cell_out = tf.add(tf.matmul(input_x, weight), bias)
            if activation_function is not None:
                cell_out_preDrop = activation_function(cell_out)
                cell_out = tf.nn.dropout(cell_out, keep_prob)

            self.cell_out = cell_out

            tf.summary.histogram('fc_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('fc_layer/{}/bias'.format(index), bias)

    def output(self):
        return self.cell_out


def my_LeNet(input_x, input_y, keep_prob,
          img_len=32, channel_num=3, output_size=5,
          conv_featmap=[[6, 16]], fc_units=[84],
          conv_kernel_size=[[5, 5]], pooling_size=[2, 2],
          l2_norm=0.01, seed=235):
    """
        LeNet is an early and famous CNN architecture for image classfication task.
        It is proposed by Yann LeCun. Here we use its architecture as the startpoint
        for your CNN practice. Its architecture is as follow.

        input >> Conv2DLayer >> Conv2DLayer >> flatten >>
        DenseLayer >> AffineLayer >> softmax loss >> output

        Or

        input >> [conv2d-maxpooling] >> [conv2d-maxpooling] >> flatten >>
        DenseLayer >> AffineLayer >> softmax loss >> output

        http://deeplearning.net/tutorial/lenet.html

        TODO:
            1) Probably going to run into sizing issues when you go from pooling to different layers
                as this will change size so you need to do some checks before running to be sure
                the sizes will match up
    """
    # indices in conv_featmap and conv_kernel_size need to match up.
    # What we should do is change them though to be lists of lists
    # so we can have the patter [[conv -> relu]*N -> Pool]*M where 
    # len(pooling_size) = M = len(conv_kernel_size) = len(conv_featmap) 
    # and len(conv_featmap[i]) == len(conv_kernel_size[i]) and can be variable.
    # each inner list in conv_* represents how many conv relus to apply.

    assert len(conv_featmap) == len(conv_kernel_size) and len(conv_featmap) == len(pooling_size)
    for idx, convReluKernels in enumerate(conv_kernel_size):
        assert len(conv_kernel_size[idx]) == len(conv_featmap[idx])


    conv_w = []
    currInputConv = input_x
    currChannelNum = channel_num
    layerIdx = 0
    for N_i, _ in enumerate(pooling_size):# iterate through all outer pooling layers which follow [conv -> relu]*N_i
        for M_i, _ in enumerate(conv_featmap[N_i]):# iterate through all conv -> relu layers and apply them
            convLayer = conv_layer(input_x = currInputConv,
                              in_channel = currChannelNum,
                              out_channel = conv_featmap[N_i][M_i],
                              kernel_shape = conv_kernel_size[N_i][M_i],
                              rand_seed=seed,
                              index = layerIdx)
            currInputConv = convLayer.output()
            currChannelNum = conv_featmap[N_i][M_i]
            conv_w.append(convLayer.weight)
            layerIdx += 1
        convLayer = norm_layer(convLayer.output())
        pooling_layer = max_pooling_layer(input_x=convLayer.output(),
                                        k_size=pooling_size[N_i],
                                        padding="VALID")
        print("pooling layer shape {} at pool {}".format(pooling_layer.output().get_shape(), N_i))
        currInputConv = pooling_layer.output()# TODO this seems like it'll work but let's see





    # conv layer

    # conv_layer_0 = conv_layer(input_x=input_x,
    #                           in_channel=channel_num,
    #                           out_channel=conv_featmap[0],
    #                           kernel_shape=conv_kernel_size[0],
    #                           rand_seed=seed)

    # pooling_layer_0 = max_pooling_layer(input_x=conv_layer_0.output(),
    #                                     k_size=pooling_size[0],
    #                                     padding="VALID")

    # flatten
    pool_shape = pooling_layer.output().get_shape()
    img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(pooling_layer.output(), shape=[-1, img_vector_length])

    print("Input shape for fully connected layer: {}".format(flatten.get_shape()))
    fc_w = []
    currInputFC = flatten
    currInputSize = img_vector_length
    
    
    #keep_prob = tf.placeholder(tf.float32)
    #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    
    for idx, layerSize in enumerate(fc_units):
        if (idx + 1) == len(fc_units):
            # reached final layer so don't use an activation function
            fcLayer = fc_layer(input_x=currInputFC,
                              in_size=currInputSize,
                              out_size=output_size,
                              rand_seed=seed,
                              activation_function=None,
                              index=idx,
                              keep_prob = keep_prob)
        else:
            fcLayer = fc_layer(input_x=currInputFC,
                              in_size=currInputSize,
                              out_size=layerSize,
                              rand_seed=seed,
                              activation_function=tf.nn.relu,
                              index=idx,
                              keep_prob = keep_prob)
            #fcLayer = tf.nn.dropout(fcLayer_predrop, keep_prob)
        currInputFC = fcLayer.output()
        currInputSize = layerSize
        fc_w.append(fcLayer.weight)
    # fc layer
    # fc_layer_0 = fc_layer(input_x=flatten,
    #                       in_size=img_vector_length,
    #                       out_size=fc_units[0],
    #                       rand_seed=seed,
    #                       activation_function=tf.nn.relu,
    #                       index=0)

    # fc_layer_1 = fc_layer(input_x=fc_layer_0.output(),
    #                       in_size=fc_units[0],
    #                       out_size=output_size,
    #                       rand_seed=seed,
    #                       activation_function=None,
    #                       index=1)

    # saving the parameters for l2_norm loss
    # conv_w = [conv_layer_0.weight]
    # fc_w = [fc_layer_0.weight, fc_layer_1.weight]

    # loss
    with tf.name_scope("loss"):
        l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])
        l2_loss += tf.reduce_sum([tf.norm(w, axis=[-2, -1]) for w in conv_w])

        label = tf.one_hot(input_y, output_size)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=fcLayer.output()),
            name='cross_entropy')
        loss = tf.add(cross_entropy_loss, l2_norm * l2_loss, name='loss')

        tf.summary.scalar('LeNet_loss', loss)

    return fcLayer.output(), loss


def cross_entropy(output, input_y, nclasses = 5):
    with tf.name_scope('cross_entropy'):
        label = tf.one_hot(input_y, nclasses)
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output))

    return ce


def train_step(loss, learning_rate=1e-3):
    with tf.name_scope('train_step'):
        step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        #step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
        #step = tf.train.RMSPropOptimizer(learning_rate = learning_rate).minimize(loss)

    return step


def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('LeNet_error_num', error_num)
    return error_num


# training function for the LeNet model
def my_training(X_train, y_train, X_val, y_val, outputSize, 
             conv_featmap=[6],
             fc_units=[84],
             conv_kernel_size=[5],
             pooling_size=[2],
             l2_norm=0.01,
             seed=235,
             learning_rate=1e-2,
             epoch=20,
             batch_size=245,
             verbose=False,
             pre_trained_model=None,
             keepProbVal = 0.5,
             imageReshapeSize = 64):
    print("Building my LeNet. Parameters: ")
    print("conv_featmap={}".format(conv_featmap))
    print("fc_units={}".format(fc_units))
    print("conv_kernel_size={}".format(conv_kernel_size))
    print("pooling_size={}".format(pooling_size))
    print("l2_norm={}".format(l2_norm))
    print("seed={}".format(seed))
    print("learning_rate={}".format(learning_rate))

    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None,imageReshapeSize,imageReshapeSize, 3], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)
        keep_prob = tf.placeholder(tf.float32)
    #print("input shapes after resizing: {}".format(X_train.shape))
    output, loss = my_LeNet(xs, ys,
                         img_len=X_train.shape[1],
                         channel_num=3,
                         output_size=outputSize,
                         conv_featmap=conv_featmap,
                         fc_units=fc_units,
                         conv_kernel_size=conv_kernel_size,
                         pooling_size=pooling_size,
                         l2_norm=l2_norm,
                         seed=seed,
                         keep_prob = keep_prob)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    step = train_step(loss)
    eve = evaluate(output, ys)

    iter_total = 0
    best_acc = 0
    cur_model_name = 'lenet_{}'.format(int(time.time()))

    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        X_val = tf.image.resize_images(X_val, [imageReshapeSize, imageReshapeSize])
        X_val = X_val.eval(session=sess)
        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                print("Load model Failed!")
                pass
        start = time.time()
        for epc in range(epoch):
            start = time.time()
            print("epoch {}".format(epc + 1))

            for itr in range(iters):
                iter_total += 1

                training_batch_x = X_train[itr * batch_size: (1 + itr) * batch_size]
                training_batch_y = y_train[itr * batch_size: (1 + itr) * batch_size]
                training_batch_x = tf.image.resize_images(training_batch_x, [imageReshapeSize, imageReshapeSize])
                training_batch_x = training_batch_x.eval(session = sess)
                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x, ys: training_batch_y, 
                                                               keep_prob: keepProbVal})

                if iter_total % 100 == 0:
                    # do validation
                    valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val, ys: y_val,
                                                                               keep_prob: 1.0})
                    valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
                    if verbose:
                        print('{}/{} loss: {} validation accuracy : {}%'.format(
                            batch_size * (itr + 1),
                            X_train.shape[0],
                            cur_loss,
                            valid_acc))

                    # save the merge result summary
                    writer.add_summary(merge_result, iter_total)

                    # when achieve the best validation accuracy, we store the model paramters
                    if valid_acc > best_acc:
                        print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                        best_acc = valid_acc
                        saver.save(sess, 'model/{}'.format(cur_model_name))
            end = time.time()
            print("epoch time {}".format(end - start))
    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))










def my_training_task4(X_train, y_train, X_val, y_val, outputSize,
             conv_featmap=[6],
             fc_units=[84],
             conv_kernel_size=[5],
             pooling_size=[2],
             l2_norm=0.01,
             seed=235,
             learning_rate=1e-2,
             epoch=20,
             batch_size=245,
             verbose=False,
             pre_trained_model=None,
             keepProbVal = 0.5):
    print("Building my LeNet. Parameters: ")
    print("conv_featmap={}".format(conv_featmap))
    print("fc_units={}".format(fc_units))
    print("conv_kernel_size={}".format(conv_kernel_size))
    print("pooling_size={}".format(pooling_size))
    print("l2_norm={}".format(l2_norm))
    print("seed={}".format(seed))
    print("learning_rate={}".format(learning_rate))

    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)
        keep_prob = tf.placeholder(tf.float32)

    output, loss = my_LeNet(xs, ys,
                         img_len=32,
                         channel_num=3,
                         output_size=outputSize,
                         conv_featmap=conv_featmap,
                         fc_units=fc_units,
                         conv_kernel_size=conv_kernel_size,
                         pooling_size=pooling_size,
                         l2_norm=l2_norm,
                         seed=seed,
                         keep_prob = keep_prob)


    # TODO here you should just call
    myGenTrans = ImageGenerator(x = X_train, y = y_train)
    myGenTrans.translate(shift_height = 3, shift_width = 3) 
    myGenTrans = myGenTrans.next_batch_gen(batch_size = batch_size)

    myGenRot = ImageGenerator(x = X_train, y = y_train)
    myGenRot.rotate(angle = 3)
    myGenRot = myGenRot.next_batch_gen(batch_size = batch_size)

    myGenFlipH = ImageGenerator(x = X_train, y = y_train)
    myGenFlipH.flip(mode = 'h')
    myGenFlipH = myGenFlipH.next_batch_gen(batch_size = batch_size)

    myGenFlipV = ImageGenerator(x = X_train, y = y_train)
    myGenFlipV.flip(mode = 'v')
    myGenFlipV = myGenFlipV.next_batch_gen(batch_size = batch_size)

    myGenFlipHV = ImageGenerator(x = X_train, y = y_train)
    myGenFlipHV.flip(mode = 'HV')
    myGenFlipHV = myGenFlipHV.next_batch_gen(batch_size = batch_size)

    myGenNoise = ImageGenerator(x = X_train, y = y_train)
    myGenNoise.add_noise(portion = 0.5, amplitude = 0.1)
    myGenNoise = myGenNoise.next_batch_gen(batch_size = batch_size)

    myGenNone = ImageGenerator(x = X_train, y = y_train)
    myGenNone = myGenNone.next_batch_gen(batch_size = batch_size)

    generatePhaseList = [myGenNone, myGenNoise, myGenFlipHV, myGenFlipV, myGenFlipH, myGenRot, myGenTrans]



    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    step = train_step(loss)
    eve = evaluate(output, ys)

    iter_total = 0
    best_acc = 0
    cur_model_name = 'kaggle_lenet_{}'.format(int(time.time()))

    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                print("Load model Failed!")
                pass
        start = time.time()
        for epc in range(epoch):
            start = time.time()
            print("epoch {}".format(epc + 1))


            for genPhase in generatePhaseList:
                for itr in range(iters):
                    iter_total += 1
                    # if batchPhase == "None":
                    training_batch_x, training_batch_y = next(genPhase)
                    # elif batchPhase == "translate":
                    # elif batchPhase == "rotate":
                    # elif batchPhase == "flipH":
                    # elif batchPhase == "flipV":
                    # elif batchPhase == "flipHV":
                    # elif batchPhase == "noise":
                    # training_batch_x = X_train[itr * batch_size: (1 + itr) * batch_size]
                    # training_batch_y = y_train[itr * batch_size: (1 + itr) * batch_size]

                    _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x, ys: training_batch_y, 
                                                                   keep_prob: keepProbVal})

                    if iter_total % 100 == 0:
                        # do validation
                        valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val, ys: y_val,
                                                                                   keep_prob: 1.0})
                        valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
                        if verbose:
                            print('{}/{} loss: {} validation accuracy : {}%'.format(
                                batch_size * (itr + 1),
                                X_train.shape[0],
                                cur_loss,
                                valid_acc))

                        # save the merge result summary
                        writer.add_summary(merge_result, iter_total)

                        # when achieve the best validation accuracy, we store the model paramters
                        if valid_acc > best_acc:
                            print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                            best_acc = valid_acc
                            saver.save(sess, 'model/{}'.format(cur_model_name))
            end = time.time()
            print("epoch time {}".format(end - start))
    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))