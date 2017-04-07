import os
import sys
import gzip
import pickle
import logging
import argparse

import gym

from models.models import (
    MODELS_LIST,
    A8DoubleQNet
)
from models.base import (
    LinearValueFunctionApprox, 
    HiddenValueFunctionApprox
)
from core.buffers import HistoryBuffer
from core.qlearning import (
    do_batch_qlearning, 
    do_online_qlearning, 
    do_online_double_qlearning
)

from core.running import run_multiple_trials

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', nargs='?', type=str,
                      help='Select model to train')
    parser.add_argument('--train', nargs='?', const=True, type=bool,
                  default=False,
                  help='If true, train model with fixed learning rate.')

    FLAGS, _ = parser.parse_known_args()

    dumps_dir = os.path.join(
        os.path.dirname(__file__), 'dumps')      
    if not os.path.exists(dumps_dir):
        os.mkdir(dumps_dir)


    env = gym.make('CartPole-v0')
    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]

    if FLAGS.model == 'A31':
        if FLAGS.train:
            history_buffer = HistoryBuffer(env, 2000, 300, 100)
            model = LinearValueFunctionApprox(4, 2)
            run_multiple_trials(env, history_buffer, model, learning_rates, 10, dumps_dir)

    elif FLAGS.model == 'A32':
        if FLAGS.train:
            history_buffer = HistoryBuffer(env, 2000, 300, 100)
            model = HiddenValueFunctionApprox(4, 2, 100)
            run_multiple_trials(env, history_buffer, model, learning_rates, 10, dumps_dir)

    elif FLAGS.model in MODELS_LIST:
        if FLAGS.train:
            main_model = MODELS_LIST[FLAGS.model]
            do_online_qlearning(env, 
                                model=main_model.model, 
                                learning_rate=main_model.learning_rate,
                                epsilon_s=main_model.epsilon_s, 
                                num_episodes=main_model.num_episodes,
                                len_episodes=main_model.len_episodes,
                                target_model=main_model.target_model,
                                replay_buffer=main_model.replay_buffer,
                                dpaths=)

    elif FLAGS.model == 'A8':
        if FLAGS.train:
            main_model = A8DoubleQNet()
            do_online_double_qlearning(env, 
                                    model_1=main_model.model_1, 
                                    model_2=main_model.model_2, 
                                    learning_rate=main_model.learning_rate,
                                    epsilon_s=main_model.epsilon_s, 
                                    num_episodes=main_model.num_episodes,
                                    len_episodes=main_model.len_episodes,
                                    target_model=main_model.target_model,
                                    replay_buffer=main_model.replay_buffer,
                                    dpaths=)

    else:
        print('No corresponding models')



