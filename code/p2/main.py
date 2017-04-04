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

ENVS = {
    'pong': {
        'gym_name': 'Pong-v3',
        'learning_rate': 0.0001,
    },
    'pacman': {
        'gym_name': 'MsPacman-v3',
        'learning_rate': 0.0001, 
    },
    'boxing': {
        'gym_name': 'Boxing-v3',
        'learning_rate': 0.0001,
    }
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', nargs='?', type=str,
                      help='Select environment to train')
    parser.add_argument('--train', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, train model with fixed learning rate.')

    FLAGS, _ = parser.parse_known_args()

    try:
        tenv = ENVS[FLAGS.env]
    except KeyError:
        print('Env does not exist.')
        raise

    dumps_dir = os.path.join(
        os.path.dirname(__file__), 'dumps', FLAGS.env)      
    if not os.path.exists(dumps_dir):
        os.mkdir(dumps_dir)

    losses_file = os.path.join(
        dumps_dir, 'losses.csv')

    results_file = os.path.join(
            dumps_dir, 'results.csv')

    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]

    #learning_rates = [0.01, 0.1]

    env = gym.make(tenv['gym_name'])
    test_env = gym.make(tenv['gym_name'])

    if FLAGS.train:
        epsilon_s = { 'start': 0.5, 'end': 0.005, 'decay': 2000 }
        loss, means = do_online_qlearning(env, test_env,
                            model=AtariModel(env.action_space.n), 
                            learning_rate=0.0001,
                            epsilon_s=epsilon_s, 
                            target_model=AtariModel(env.action_space.n, varscope='target'),
                            replay_buffer=ExperienceReplayBuffer(500000, 64),
                            dpaths=dumps_dir)

        np.savetxt(losses_file, loss, delimiter=',')
        np.savetxt(results_file, means, delimiter=',')


