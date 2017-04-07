import tensorflow as tf
from models.utils import weights, biases

from core.buffers import ExperienceReplayBuffer
from models.base import HiddenValueFunctionApprox

class A4Hidden100Units:
    model = HiddenValueFunctionApprox(4, 2, 100)
    replay_buffer = None
    target_model = None 
    learning_rate = 0.001 
    num_episodes = 2001
    len_episodes = 300
    epsilon_s = { 'start': 0.5, 'end': 0.005, 'decay': 2000 }

class A5Hidden30Units:
    model = HiddenValueFunctionApprox(4, 2, 30)
    replay_buffer = None
    target_model = None 
    learning_rate = 0.001 
    num_episodes = 2001
    len_episodes = 300
    epsilon_s = { 'start': 0.5, 'end': 0.005, 'decay': 2000 }

class A5Hidden1000Units:
    model = HiddenValueFunctionApprox(4, 2, 1000)
    replay_buffer = None
    target_model = None 
    learning_rate = 0.001 
    num_episodes = 2001
    len_episodes = 300
    epsilon_s = { 'start': 0.5, 'end': 0.005, 'decay': 2000 }

class A6Hidden100UnitsReplayBuffer:
    model = HiddenValueFunctionApprox(4, 2, 100)
    replay_buffer = ExperienceReplayBuffer(1000, 100)
    target_model = None 
    learning_rate = 0.001 
    num_episodes = 2001
    len_episodes = 300
    epsilon_s = { 'start': 0.5, 'end': 0.005, 'decay': 2000 }

class A7Hidden100UnitsReplayBufferTargetNet:
    model = HiddenValueFunctionApprox(4, 2, 100)
    replay_buffer = ExperienceReplayBuffer(1000, 100)
    target_model = HiddenValueFunctionApprox(4, 2, 100, varscope='target') 
    learning_rate = 0.001 
    num_episodes = 2001
    len_episodes = 300
    epsilon_s = { 'start': 0.5, 'end': 0.005, 'decay': 2000 }

class A8DoubleQNet:
    model_1 = HiddenValueFunctionApprox(4, 2, 100, varscope='1')
    model_2 = HiddenValueFunctionApprox(4, 2, 100, varscope='2')
    replay_buffer = ExperienceReplayBuffer(1000, 100)
    target_model = HiddenValueFunctionApprox(4, 2, 100, varscope='target') 
    learning_rate = 0.001 
    num_episodes = 2001
    len_episodes = 300
    epsilon_s = { 'start': 0.5, 'end': 0.005, 'decay': 2000 }


MODELS_LIST = {
    'A4': A4Hidden100Units,
    'A51': A5Hidden30Units,
    'A52': A5Hidden1000Units,
    'A6': A6Hidden100UnitsReplayBuffer,
    'A7': A7Hidden100UnitsReplayBufferTargetNet,
}




        

