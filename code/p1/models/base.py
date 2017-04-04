import tensorflow as tf
from models.utils import weights, biases


class LinearValueFunctionApprox(object):

    def __init__(self, dim_in, dim_out):
        super(LinearValueFunctionApprox, self).__init__()
        self.name = 'linear_%sx%s' % (dim_in, dim_out)
        self.dim_in = dim_in
        self.dim_out = dim_out

    def graph(self, state):

        with tf.variable_scope('linear'):
            W_lin = weights([self.dim_in, self.dim_out], 'lin')
            b_lin = biases([self.dim_out], 'lin')
            out = tf.matmul(state, W_lin) + b_lin

        return out


class HiddenValueFunctionApprox(object):

    def __init__(self, dim_in, dim_out, units=100, varscope=''):
        super(HiddenValueFunctionApprox, self).__init__()
        self.name = 'hidden_%sx%sx%s' % (dim_in, dim_out, units)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.units = units
        self.varscope = '%s_' % varscope if varscope != '' else ''

    def graph(self, state):

        with tf.variable_scope('hidden'):
            W_hidden = weights([self.dim_in, self.units], '%shidden' % self.varscope)
            b_hidden = biases([self.units], '%shidden' % self.varscope)
            out_hidden = tf.nn.relu(
                            tf.matmul(state, W_hidden) + b_hidden)

        with tf.variable_scope('linear'):
            W_lin = weights([self.units, self.dim_out], '%slin' % self.varscope)
            b_lin = biases([self.dim_out], '%slin' % self.varscope)
            out = tf.matmul(out_hidden, W_lin) + b_lin

        return out