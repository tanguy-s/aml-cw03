import sys
import time

import numpy as np
import tensorflow as tf

from core.utils import evaluate, reward_value

GAMMA = 0.99

def do_testing(env, 
                model, 
                num_episodes, 
                len_episodes,
                dpaths=None):

    tf.reset_default_graph()

    # Create placeholders
    states_pl = tf.placeholder(tf.float32, shape=(None, 4), name='states')
    actions_pl= tf.placeholder(tf.int32, shape=(None,), name='actions')
    targets_pl = tf.placeholder(tf.float32, shape=(None,), name='targets')

    # Value function approximator network
    q_output = model.graph(states_pl)

    # Compute Q from current q_output and one hot actions
    Q = tf.reduce_sum(
            tf.multiply(q_output, 
                tf.one_hot(actions_pl, 2, dtype=tf.float32)
            ), axis=1)

    # Loss operation 
    loss_op = tf.reduce_mean(tf.square(targets_pl - Q) / 2)

    # Prediction Op
    prediction = tf.argmax(q_output, 1)

    # Model Saver
    saver = tf.train.Saver()

    # init all variables
    init_op = tf.global_variables_initializer()

    # Start Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Start Session
    with tf.Session(config=config) as sess:

        if dpaths is not None:
            new_saver = tf.train.import_meta_graph('%s.meta' % dpaths[1])
            new_saver.restore(sess, tf.train.latest_checkpoint(dpaths[0]))

            means, stds = evaluate(env, sess, prediction, 
                                    states_pl, num_episodes, len_episodes, GAMMA, False)

            # Save means
            print(means)