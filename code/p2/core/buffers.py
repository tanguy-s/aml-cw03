import random

import numpy as np



class ExperienceReplayBuffer(object):

    """
    Store experience replay:
    {(Si, Ai, Ri+1, Si+1)}
    One transition experience is set of 
    state, action, reward, next_state

    When the buffer is ready replace act like a 
    queue and replace oldest experiences
    """

    def __init__(self, buffer_size, batch_size):
        super(ExperienceReplayBuffer, self).__init__()
        self.transitions = list()
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def add(self, transition):
        if len(self.transitions) + 1 >= self.buffer_size:
            self.transitions[0:(1 + len(self.transitions)) - self.buffer_size] = []
        self.transitions.append(transition)

    @property
    def complete(self):
        return True if len(self.transitions) == self.buffer_size else False

    @property
    def ready(self):
        return True if len(self.transitions) > self.batch_size else False

    def get_rand_transitions(self):
        return random.sample(self.transitions, self.batch_size)


        
