import numpy as np


def reward_value(done):
    # Reward function: 0 in all states except 
    # if episode is done -1
    return -1 if done else 0


def evaluate(env,
                sess, 
                prediction_op, 
                states_pl, 
                num_episodes, 
                len_episodes, 
                gamma, 
                silent=False):

    res = np.zeros([num_episodes, 2])
    for i_episode in range(num_episodes):

        # Reset environement variables
        state = env.reset()
        done = False
        retval = 0 # Value function

        for t in range(len_episodes):

            actions = sess.run(prediction_op, feed_dict={
                    states_pl: state.reshape(-1,4)
                })
            state, _, done, info = env.step(actions[0])
            reward = reward_value(done)
            retval += pow(gamma, t)*reward

            if done:
                res[i_episode][0] = t+1
                res[i_episode][1] = retval
                break

    means = np.mean(res, axis=0)
    stds = np.std(res, axis=0)

    if not silent:
        print('# Evaluation')
        print('- Episode length stats:\n Mean: %f std: %f' % 
            (means[0], stds[0]))
        print('- Return from initial stats:\n Mean: %f std: %f' % 
            (means[1], stds[1]))

    return means, stds

