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
from core.running import (
    run_multiple_trials_batch, 
    run_multiple_trials_online
)

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

    dpaths = [os.path.join(dumps_dir, FLAGS.model), 
                os.path.join(dumps_dir, FLAGS.model, FLAGS.model)]
    if not os.path.exists(dpaths[0]):
        os.mkdir(dpaths[0])

    if FLAGS.model == 'A31':

        if FLAGS.train:
            history_buffer = HistoryBuffer(env, 2000, 300, 100)
            model = LinearValueFunctionApprox(4, 2)
            run_multiple_trials_batch(env, history_buffer, model, learning_rates, 10, dpaths)

        elif FLAGS.test:
            pass

    elif FLAGS.model == 'A32':

        if FLAGS.train:
            history_buffer = HistoryBuffer(env, 2000, 300, 100)
            model = HiddenValueFunctionApprox(4, 2, 100)
            run_multiple_trials_batch(env, history_buffer, model, learning_rates, 10, dpaths)

        elif FLAGS.test:
            pass

    elif FLAGS.model in MODELS_LIST:

        if FLAGS.train:
            main_model = MODELS_LIST[FLAGS.model]
            run_multiple_trials_online(env, main_model, 20, dpaths)

        elif FLAGS.test:
            pass

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
                                    dpaths=dpaths)

        elif FLAGS.test:
            pass

    else:
        print('No corresponding models')

# nohup python main.py -m A31 --train &> log_a31.out &
# nohup python main.py -m A32 --train &> log_a32.out &
# nohup python main.py -m A4 --train &> log_a4.out &
# nohup python main.py -m A51 --train &> log_a51.out &
# nohup python main.py -m A52 --train &> log_a52.out &
# nohup python main.py -m A6 --train &> log_a6.out &
# nohup python main.py -m A8 --train &> log_a8.out &


