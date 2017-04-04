import tensorflow as tf


def weights(shape, name):
    return tf.get_variable('W_%s' % name, shape, 
				initializer=tf.random_normal_initializer(stddev=0.01))

def biases(shape, name):
    return tf.get_variable('b_%s' % name, shape, 
    			initializer=tf.constant_initializer(0.1))