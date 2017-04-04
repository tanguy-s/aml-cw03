import tensorflow as tf
from models.utils import weights, biases



class AtariModel(object):

    def __init__(self, dim_out, varscope=''):
        super(AtariModel, self).__init__()
        self.name = 'attari_%s' % (dim_out)
        self.dim_out = dim_out
        self.varscope = '%s_' % varscope if varscope != '' else ''

    def graph(self, state):

        with tf.variable_scope('conv1'):
            W_conv1 = weights([6, 6, 4, 16], '%sconv1' % self.varscope)
            b_conv1 = biases([16], '%sconv1' % self.varscope)
            out_conv1 = tf.nn.relu(
                            tf.nn.conv2d(state, W_conv1, 
                                strides=[1, 2, 2, 1], padding='SAME') + b_conv1)


        with tf.variable_scope('conv2'):
            W_conv2 = weights([4, 4, 16, 32], '%sconv2' % self.varscope)
            b_conv2 = biases([32], '%sconv2' % self.varscope)
            out_conv2 = tf.nn.relu(
                            tf.nn.conv2d(out_conv1, W_conv2, 
                                strides=[1, 2, 2, 1], padding='SAME') + b_conv2)
            # Flatten conv 1 output
            o_shp = out_conv2.get_shape().as_list()
            conv2_dim = o_shp[1] * o_shp[2] * o_shp[3]
            out_conv2_flat = tf.reshape(out_conv2, [-1, conv2_dim])

        with tf.variable_scope('fully_connected'):
            W_fc = weights([conv2_dim, 256], '%sfc' % self.varscope)
            b_fc = biases([256], '%sfc' % self.varscope)
            out_fc = tf.nn.relu(tf.matmul(out_conv2_flat, W_fc) + b_fc)

        with tf.variable_scope('linear'):
            W_lin = weights([256, self.dim_out], '%slin' % self.varscope)
            b_lin = biases([self.dim_out], '%slin' % self.varscope)
            out = tf.matmul(out_fc, W_lin) + b_lin

        return out