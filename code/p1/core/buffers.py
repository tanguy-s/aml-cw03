import random

import numpy as np

from core.utils import reward_value


class HistoryBuffer(object):

    def __init__(self, env, num_episodes, len_episodes, batch_size):
        super(HistoryBuffer, self).__init__()
        self.env = env
        self.num_episodes = num_episodes
        self.len_episodes = len_episodes
        self.batch_size = batch_size
        self.states, self.actions, self.rewards, self.term_state = None, None, None, None

        # Properties
        self.num_batch = int(num_episodes / self.batch_size)
        self.curr_batch = 0

        self.get_history_buffer()

        print('# Done building history buffer.')
        print('- Buffer size is %d number of batches: %d' % 
                (len(self.states), self.num_batch))


    def next_batch(self):
        # Return previous states, next states, actions, rewards
        b_start = self.curr_batch * self.batch_size
        b_end = (self.curr_batch + 1) * self.batch_size

        if self.curr_batch < self.num_batch:
            self.curr_batch += 1
        else:
            self.curr_batch = 0

        return self.states[b_start:b_end], \
                self.states[b_start + 1:b_end + 1], \
                self.actions[b_start:b_end], \
                self.rewards[b_start:b_end], \
                self.term_state[b_start:b_end]


    def get_history_buffer(self):
        # Arrays [num_episodes x length of episode]
        # The length of episode is unknown
        # Using python list and convert to numpy after
        states = list()
        actions = list()
        rewards = list()
        term_state = list()

        for i_episode in range(self.num_episodes):

            # Reset environement variables
            state = self.env.reset()
            done = False

            # Save current episode's values
            epi_sta = list()
            epi_act = list()
            epi_rwd = list()
            epi_tst = list()

            for t in range(self.len_episodes):
                #self.env.render()

                # Uniform sample of action in action space 
                action = self.env.action_space.sample()
                # Save current state & action
                epi_sta.append(state)
                epi_act.append(action)

                # Start next step
                state, _, done, info = self.env.step(action)
                reward = reward_value(done)

                # Save current reward
                epi_rwd.append(reward)
                # Save terminal state
                epi_tst.append(done)

                if done:
                    states.append(np.array(epi_sta))
                    actions.append(np.array(epi_act))
                    rewards.append(np.array(epi_rwd))
                    term_state.append(np.array(epi_tst))
                    break

            if i_episode % 500 == 0:
                print('Replay buffer size %d' % i_episode)

        self.states = np.concatenate(
            [states[i] for i in range(self.num_episodes)], axis=0)
        self.actions = np.concatenate(
            [actions[i] for i in range(self.num_episodes)], axis=0)
        self.rewards = np.concatenate(
            [rewards[i] for i in range(self.num_episodes)], axis=0)
        self.term_state = np.concatenate(
            [term_state[i] for i in range(self.num_episodes)], axis=0)


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
    def ready(self):
        return True if len(self.transitions) == self.buffer_size else False

    def next_transitions(self):
        batch_transitions = random.sample(self.transitions, self.batch_size)
        states = np.stack(
            [batch_transitions[i][0] for i in range(len(batch_transitions))], axis=0).reshape([-1,4]).astype('float32')
        actions = np.stack(
            [batch_transitions[i][1] for i in range(len(batch_transitions))], axis=0).reshape([-1])
        reward = np.stack(
            [batch_transitions[i][2] for i in range(len(batch_transitions))], axis=0).reshape([-1])
        next_state = np.stack(
            [batch_transitions[i][3] for i in range(len(batch_transitions))], axis=0).reshape([-1,4]).astype('float32')
        term_state = np.stack(
            [batch_transitions[i][4] for i in range(len(batch_transitions))], axis=0).reshape([-1])

        return states, actions, reward, next_state, term_state


        
