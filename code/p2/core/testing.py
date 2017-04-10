import os
import sys
import time

import numpy as np
import tensorflow as tf

from core.utils import evaluate, reward_value, do_obs_processing, reward_clip

GAMMA = 0.99
TEST_STEPS = 100
FRAME_WIDTH = 84
FRAME_HEIGHT = 84
FRAME_BUFFER_SIZE = 4


def do_testing(env, model, target_model=None, dpaths=None, render=False, num_episodes=100):

    tf.reset_default_graph()


    # Create placeholders
    states_pl = tf.placeholder(tf.float32, 
        shape=(None, FRAME_WIDTH, FRAME_HEIGHT, 4), name='states')
    actions_pl= tf.placeholder(tf.int32, shape=(None), name='actions')
    targets_pl = tf.placeholder(tf.float32, shape=(None), name='targets')

    # Value function approximator network
    q_output = model.graph(states_pl)

    # Build target network
    q_target_net = target_model.graph(states_pl)

    # Compute Q from current q_output and one hot actions
    Q = tf.reduce_sum(
            tf.multiply(q_output, 
                tf.one_hot(actions_pl, env.action_space.n, dtype=tf.float32)
            ), axis=1)

    # Loss operation 
    loss_op = tf.reduce_mean(tf.square(targets_pl - Q) / 2)

    # Prediction Op
    prediction = tf.argmax(q_output, 1)

    # Model Saver
    saver = tf.train.Saver()

    # init all variables
    init_op = tf.global_variables_initializer()

    # Limit memory usage for multiple training at same time
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.33

    # Start Session
    with tf.Session(config=config) as sess:

        if dpaths is not None:
            new_saver = tf.train.import_meta_graph(dpaths[1])
            new_saver.restore(sess, tf.train.latest_checkpoint(dpaths[0]))

            means, stds = evaluate(env, sess, prediction, 
                                    states_pl, num_episodes, GAMMA, False, render)

            # Save means
            print(means)