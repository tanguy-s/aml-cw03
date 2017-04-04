import warnings

import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize


FRAME_WIDTH = 84
FRAME_HEIGHT = 84
FRAME_BUFFER_SIZE = 4


def reward_clip(reward):
    return np.clip(reward, -1, 1)

def reward_value(done):
    # Reward function: 0 in all states except 
    # if episode is done -1
    return -1 if done else 0

def evaluate(env,
                sess, 
                prediction_op, 
                states_pl, 
                num_episodes, 
                gamma, 
                silent=False):

    

    res = np.zeros([num_episodes, 2])
    for i_episode in range(num_episodes):
        #Observation Buffer
        observation_buffer = list()

        # Reset environement variables
        observation = env.reset()
        done = False
        retval = 0 # Value function
        score = 0
        t = 0

        while not done:
            t += 1            

            # Stack observations in buffer of 4
            if len(observation_buffer) < FRAME_BUFFER_SIZE:

                observation_buffer.append(
                    do_obs_processing(observation, FRAME_WIDTH, FRAME_HEIGHT))

                # Collect next observation with uniformly random action
                a_rnd = env.action_space.sample()
                observation, _, done, _ = env.step(a_rnd)

            # Observations buffer is ready
            else:
                # Stack observation buffer
                state = np.stack(observation_buffer, axis=-1)

                action = sess.run(prediction_op, feed_dict={
                    states_pl: state.reshape(
                        [-1, FRAME_WIDTH, FRAME_HEIGHT, FRAME_BUFFER_SIZE]).astype('float32')
                })

                # action for next observation
                observation, reward, done, info  = env.step(action[0])

                # Clip reward
                r = reward_clip(reward)

                # Return value
                retval += pow(gamma, t)*r
                score += r

                observation_buffer.append(
                    do_obs_processing(observation, FRAME_WIDTH, FRAME_HEIGHT))
                observation_buffer[0:1] = []

        res[i_episode][0] = retval
        res[i_episode][1] = score

    means = np.mean(res, axis=0)
    stds = np.std(res, axis=0)

    if not silent:
        print('# Evaluation of policy')
        print('- Return stats:\n Mean: %f std: %f' % 
            (means[0], stds[0]))
        print('- Scores stats:\n Mean: %f std: %f' % 
            (means[1], stds[1]))

    return means


def do_obs_processing(frame, width, height):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return rgb2gray(
            resize(frame, [width, height], preserve_range=True)).astype('uint8')

