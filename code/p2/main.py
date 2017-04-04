import os
import sys
import gzip
import pickle
import logging
import argparse

import gym

from models.models import AtariModel
from core.buffers import ExperienceReplayBuffer
from core.qlearning import do_online_qlearning

#from core.running import run_multiple_trials

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', nargs='?', type=str,
                      help='Select model to train')

    FLAGS, _ = parser.parse_known_args()

    dumps_dir = os.path.join(
        os.path.dirname(__file__), 'dumps')      
    if not os.path.exists(dumps_dir):
        os.mkdir(dumps_dir)

    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]

    #learning_rates = [0.01, 0.1]

    env = gym.make('Pong-v3')
    test_env = gym.make('Pong-v3')
    #history_buffer = HistoryBuffer(env, 2000, 300, 100)

    #model1 = LinearValueFunctionApprox(4, 2)
    #run_multiple_trials(env, history_buffer, model1, learning_rates, 10, dumps_dir)

    #model2 = HiddenValueFunctionApprox(4, 2, 100)
    #run_multiple_trials(env, history_buffer, model2, learning_rates, 10, dumps_dir)

    #do_batch_qlearning(env, history_buffer, model, 0.0001)


    epsilon_s = { 'start': 0.1, 'end': 0.005, 'decay': 2000 }
    # model = HiddenValueFunctionApprox(4, 2, 100)


    #do_online_qlearning(env, model, 0.001, epsilon_s, 2001, 300)

    # for name, main_model in MODELS_LIST.items():

    do_online_qlearning(env, test_env,
                        model=AtariModel(env.action_space.n), 
                        learning_rate=0.001,
                        epsilon_s=epsilon_s, 
                        target_model=AtariModel(env.action_space.n, varscope='target'),
                        replay_buffer=ExperienceReplayBuffer(500000, 32))


