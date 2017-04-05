import tensorflow as tf


def weights(shape, name):
    return tf.get_variable('W_%s' % name, shape, 
				initializer=tf.truncated_normal_initializer(
								mean=0.0, stddev=0.01, dtype=tf.float32))

def biases(shape, name):
    return tf.get_variable('b_%s' % name, shape, 
    			initializer=tf.constant_initializer(0.1))