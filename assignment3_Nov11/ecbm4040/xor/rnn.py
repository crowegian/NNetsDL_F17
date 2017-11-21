#!/usr/bin/env python
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
import collections
from tensorflow.python.util import nest
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"





_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class LSTMStateTuple(_LSTMStateTuple):
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.
  Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
  and `h` is the output.
  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, h) = self
    if c.dtype != h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype










class MyLSTMCell(RNNCell):
    """
    Your own basic LSTMCell implementation that is compatible with TensorFlow. To solve the compatibility issue, this
    class inherits TensorFlow RNNCell class.

    For reference, you can look at the TensorFlow LSTMCell source code. To locate the TensorFlow installation path, do
    the following:

    1. In Python, type 'import tensorflow as tf', then 'print(tf.__file__)'

    2. According to the output, find tensorflow_install_path/python/ops/rnn_cell_impl.py

    So this is basically rewriting the TensorFlow LSTMCell, but with your own language.

    Also, you will find Colah's blog about LSTM to be very useful:
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    """

    def __init__(self, num_units, num_proj, forget_bias=1.0, activation=None):
        """
        Initialize a class instance.

        In this function, you need to do the following:

        1. Store the input parameters and calculate other ones that you think necessary.

        2. Initialize some trainable variables which will be used during the calculation.

        :param num_units: The number of units in the LSTM cell.
        :param num_proj: The output dimensionality. For example, if you expect your output of the cell at each time step
                         to be a 10-element vector, then num_proj = 10.
        :param forget_bias: The bias term used in the forget gate. By default we set it to 1.0.
        :param activation: The activation used in the inner states. By default we use tanh.

        There are biases used in other gates, but since TensorFlow doesn't have them, we don't implement them either.
        """
        super(MyLSTMCell, self).__init__(_reuse=True)
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        self.num_units = num_units
        self.forget_bias = forget_bias
        # self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        # self.linear = None




      
        self.num_proj = num_proj
        state_is_tuple = True
        self.state_is_tuple = True
        if num_proj:
          self._state_size = (
              LSTMStateTuple(num_units, num_proj)
              if state_is_tuple else num_units + num_proj)
          self._output_size = num_proj
        else:
          self._state_size = (
              LSTMStateTuple(num_units, num_units)
              if state_is_tuple else 2 * num_units)
          self._output_size = num_units
        self.linear1 = None
        self.linear2 = None
        print("num units {}".format(self.num_units))
        print("num proj {}".format(self.num_proj))
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope, initializer = None):
          self.W1 = tf.get_variable(
            "lin_w1", [1 + self.num_proj, 4 * self.num_units],
            dtype=tf.float32,
            initializer=None)
          self.b1 = tf.get_variable(
            "lin_b1", [4 * self.num_units],
            dtype=tf.float32,
            initializer=init_ops.constant_initializer(0.0, dtype=tf.float32))


          self.W2 = tf.get_variable(
            "lin_w2", [self.num_units, self.num_proj],
            dtype=tf.float32,
            initializer=None)
    @property
    def state_size(self):
        """
        Overrides parent class method. Returns the state size of of the cell.

        state size = num_units + output_size

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        return(self._state_size)
        raise NotImplementedError('Please edit this function.')

    @property
    def output_size(self):
        """
        Overrides parent class method. Returns the output size of the cell.

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        return(self._output_size)
        raise NotImplementedError('Please edit this function.')

    def call(self, inputs, state):
        """
        Run one time step of the cell. That is, given the current inputs and the state from the last time step,
        calculate the current state and cell output.

        You will notice that TensorFlow LSTMCell has a lot of other features. But we will not try them. Focus on the
        very basic LSTM functionality.

        Hint 1: If you try to figure out the tensor shapes, use print(a.get_shape()) to see the shape.

        Hint 2: In LSTM there exist both matrix multiplication and element-wise multiplication. Try not to mix them.

        :param inputs: The input at the current time step. The last dimension of it should be 1.
        :param state:  The state value of the cell from the last time step. The state size can be found from function
                       state_size(self).
        :return: A tuple containing (output, new_state). For details check TensorFlow LSTMCell class.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        # raise NotImplementedError('Please edit this function.')
        # num_proj = self.num_units if self.num_proj is None else self.num_proj
        sigmoid = math_ops.sigmoid

        (c_prev, m_prev) = state


        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
          raise ValueError("Could not infer input size from inputs.get_shape()[-1]")



        scope = tf.get_variable_scope()
        if self.linear1 is None and self.linear2 is None:
          with tf.variable_scope(scope, initializer = None):#self._initializer) as unit_scope:
            self.linear1 = math_ops.matmul(array_ops.concat([inputs, m_prev], 1), self.W1)#math_ops.matmul(inputs, self.W1)#
            lstm_matrix = nn_ops.bias_add(self.linear1, self.b1)# this gives you all matrices you need. two matrices for each gate
            # a W and U matrix for the input and the preious hidden stat (m_prev)
            i, j, f, o = array_ops.split(
                value=lstm_matrix, num_or_size_splits=4, axis=1)
            c = (sigmoid(f + 1.0) * c_prev + sigmoid(i) *
               self._activation(j))
            print("inputs shape {}".format(inputs.get_shape()))
            print("m_prev shape {}".format(m_prev.get_shape()))
            print("i shape {}".format(i.get_shape()))
            print("j shape {}".format(j.get_shape()))
            print("f shape {}".format(f.get_shape()))
            print("o shape {}".format(o.get_shape()))
            print("c shape {}".format(c.get_shape()))
            m = sigmoid(o) * self._activation(c)
            print("m shape {}".format(m.get_shape()))
            print("W2 shape {}".format(self.W2.get_shape()))
            print("W1 shape {}".format(self.W1.get_shape()))
            if self.num_proj is not None:
              m = math_ops.matmul(m, self.W2)
              print("m proj shape {}".format(m.get_shape()))
            self.linear2 = 1# not None.... so hacky.


        new_state = (LSTMStateTuple(c, m) if self.state_is_tuple else
                     array_ops.concat([c, m], 1))
        return m, new_state
