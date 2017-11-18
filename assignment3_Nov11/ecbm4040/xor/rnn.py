#!/usr/bin/env python
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
import collections
from tensorflow.python.util import nest

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
        self.activation = activation or math_ops.tanh
        self.linear = None




        # super(LSTMCell, self).__init__(_reuse=reuse)
        # if not state_is_tuple:
        #   logging.warn("%s: Using a concatenated state is slower and will soon be "
        #                "deprecated.  Use state_is_tuple=True.", self)
        # if num_unit_shards is not None or num_proj_shards is not None:
        #   logging.warn(
        #       "%s: The num_unit_shards and proj_unit_shards parameters are "
        #       "deprecated and will be removed in Jan 2017.  "
        #       "Use a variable scope with a partitioner instead.", self)

        # self._num_units = num_units
        # self._use_peepholes = use_peepholes
        # self._cell_clip = cell_clip
        # self._initializer = initializer
        self.num_proj = num_proj
        # self._proj_clip = proj_clip
        # self._num_unit_shards = num_unit_shards
        # self._num_proj_shards = num_proj_shards
        # self._forget_bias = forget_bias
        state_is_tuple = True
        self.state_is_tuple = True
        # self._activation = activation or math_ops.tanh
        # if num_proj:
        #     self.output_size = num_proj
        # else:
        #     self.output_size = num_units
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
        # if self._use_peepholes:
        #   self._w_f_diag = None
        #   self._w_i_diag = None
        #   self._w_o_diag = None
        # raise NotImplementedError('Please edit this function.')
        if self.linear1 is None:
          scope = tf.get_variable_scope()
          with tf.variable_scope(
              scope, initializer = None):#self._initializer) as unit_scope:
            # if self._num_unit_shards is not None:
            #   unit_scope.set_partitioner(
            #       partitioned_variables.fixed_size_partitioner(
            #           self._num_unit_shards))
            # print(scope)
            # 1/0
            # print(scope.name)
            # print("ok")
            # print(tf.contrib.framework.get_name_scope())
            # TODO: TRY CALLING THIS STUFF IN INIT INSTEAD OF CALL. MAYBE THINGS WILL WORK. MAYBE NOT
            self.linear1 = _Linear([inputs, m_prev], 4 * self.num_units, True)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        lstm_matrix = self.linear1([inputs, m_prev])
        i, j, f, o = array_ops.split(
            value=lstm_matrix, num_or_size_splits=4, axis=1)


        if self.num_proj is not None:
            if self.linear2 is None:
                scope = tf.get_variable_scope()
                with tf.variable_scope(scope, initializer = None):#self._initializer):
                  with tf.variable_scope("projection") as proj_scope:
                    # if self._num_proj_shards is not None:
                    #   proj_scope.set_partitioner(
                    #       partitioned_variables.fixed_size_partitioner(
                    #           self._num_proj_shards))
                    self.linear2 = _Linear(m, self.num_proj, False)


        # Diagonal connections
    # The following 2 properties are required when defining a TensorFlow RNNCell.
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
        num_proj = self.num_units if self.num_proj is None else self.num_proj
        sigmoid = math_ops.sigmoid

        if self.state_is_tuple:# TODO change this because it will always be true
          (c_prev, m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self.num_units])
            m_prev = array_ops.slice(state, [0, self.num_units], [-1, num_proj])

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
          raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # god, why have you forsaken me? some variable scope shit?

        # print(tf.contrib.framework.get_name_scope())
        # 1/0




        # if self.linear1 is None:
        #   scope = tf.get_variable_scope()
        #   with tf.variable_scope(
        #       scope, initializer = None):#self._initializer) as unit_scope:
        #     # if self._num_unit_shards is not None:
        #     #   unit_scope.set_partitioner(
        #     #       partitioned_variables.fixed_size_partitioner(
        #     #           self._num_unit_shards))
        #     # print(scope)
        #     # 1/0
        #     # print(scope.name)
        #     # print("ok")
        #     # print(tf.contrib.framework.get_name_scope())
        #     # TODO: TRY CALLING THIS STUFF IN INIT INSTEAD OF CALL. MAYBE THINGS WILL WORK. MAYBE NOT
        #     self.linear1 = _Linear([inputs, m_prev], 4 * self.num_units, True)

        # # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        # lstm_matrix = self.linear1([inputs, m_prev])
        # i, j, f, o = array_ops.split(
        #     value=lstm_matrix, num_or_size_splits=4, axis=1)
        # # Diagonal connections

        1/0
        # peephole and clip stuff. We're not doing this
        # if self._use_peepholes and not self._w_f_diag:
        #   scope = vs.get_variable_scope()
        #   with vs.variable_scope(
        #       scope, initializer=self._initializer) as unit_scope:
        #     with vs.variable_scope(unit_scope):
        #       self._w_f_diag = vs.get_variable(
        #           "w_f_diag", shape=[self._num_units], dtype=dtype)
        #       self._w_i_diag = vs.get_variable(
        #           "w_i_diag", shape=[self._num_units], dtype=dtype)
        #       self._w_o_diag = vs.get_variable(
        #           "w_o_diag", shape=[self._num_units], dtype=dtype)

        # if self._use_peepholes:
        #   c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
        #        sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
        # else:
        c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
           self.activation(j))

        # if self._cell_clip is not None:
        #   # pylint: disable=invalid-unary-operand-type
        #   c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
        #   # pylint: enable=invalid-unary-operand-type
        # if self._use_peepholes:
        #   m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
        # else:
        m = sigmoid(o) * self._activation(c)
        1/0
        if self.num_proj is not None:
          if self.linear2 is None:
            scope = tf.get_variable_scope()
            with tf.variable_scope(scope, initializer = None):#self._initializer):
              with tf.variable_scope("projection") as proj_scope:
                # if self._num_proj_shards is not None:
                #   proj_scope.set_partitioner(
                #       partitioned_variables.fixed_size_partitioner(
                #           self._num_proj_shards))
                self.linear2 = _Linear(m, self.num_proj, False)
          m = self.linear2(m)
          # print(m)
          1/0

          # if self._proj_clip is not None:
          #   # pylint: disable=invalid-unary-operand-type
          #   m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
          #   # pylint: enable=invalid-unary-operand-type

        new_state = (LSTMStateTuple(c, m) if self.state_is_tuple else
                     array_ops.concat([c, m], 1))
        return m, new_state




class _Linear(object):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of weight variable.
    dtype: data type for variables.
    build_bias: boolean, whether to build a bias variable.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Raises:
    ValueError: if inputs_shape is wrong.
  """

  def __init__(self,
               args,
               output_size,
               build_bias,
               bias_initializer=None,
               kernel_initializer=None):
    self._build_bias = build_bias

    if args is None or (nest.is_sequence(args) and not args):
      raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
      args = [args]
      self._is_sequence = False
    else:
      self._is_sequence = True

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
      if shape.ndims != 2:
        raise ValueError("linear is expecting 2D arguments: %s" % shapes)
      if shape[1].value is None:
        raise ValueError("linear expects shape[1] to be provided for shape %s, "
                         "but saw %s" % (shape, shape[1]))
      else:
        total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    scope = tf.get_variable_scope()
    print(scope.name)
    print(tf.contrib.framework.get_name_scope())
    # 1/0
    with tf.variable_scope(scope) as outer_scope:
      print(outer_scope)
      print(outer_scope.name)
      # 1/0
      # getting here and for some reason not creating this variable. the 1/0 below does not happen
      self._weights = tf.get_variable(
          _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
          dtype=dtype,
          initializer=kernel_initializer)
      print(self._weights.name)
      1/0
      if build_bias:
        with tf.variable_scope(outer_scope) as inner_scope:
          inner_scope.set_partitioner(None)
          if bias_initializer is None:
            bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
          self._biases = tf.get_variable(
              _BIAS_VARIABLE_NAME, [output_size],
              dtype=dtype,
              initializer=bias_initializer)
    1/0

  def __call__(self, args):
    if not self._is_sequence:
      args = [args]

    if len(args) == 1:
      res = math_ops.matmul(args[0], self._weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), self._weights)
    if self._build_bias:
      res = nn_ops.bias_add(res, self._biases)
    return res