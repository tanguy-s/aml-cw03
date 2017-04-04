import gym
import numpy as np
# Load environment
env = gym.make('CartPole-v0')


NUM_EPISODES = 100
EPISODE_LENGTH = 300
GAMMA = 0.99 # Discount factor

def reward_value(done):
    # Reward function: 0 in all states except 
    # if episode is done -1
    return -1 if done else 0

def run_env():
    res = np.zeros([NUM_EPISODES, 2])
    for i_episode in range(NUM_EPISODES):

        # Reset environement variables
        observation = env.reset()
        done = False
        retval = 0 # Value function

        for t in range(EPISODE_LENGTH):
            env.render()

            # Uniform sample of action in action space 
            action = env.action_space.sample()
            observation, _, done, info = env.step(action)
            reward = reward_value(done)
            retval += pow(GAMMA, t)*reward

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                print('Return value is %f' % retval)
                res[i_episode][0] = t+1
                res[i_episode][1] = retval
                break

    means = np.mean(res, axis=0)
    stds = np.std(res, axis=0)
    print('## Q.1')
    for i in range(3):
        print('- Run #%d Episode length: %d Return value: %f' % 
            ((i+1), res[i][0], res[i][1]))
    print('## Q.2')
    print('- Episode length stats:\n Mean: %f std: %f' % 
        (means[0], stds[0]))
    print('- Return from initial stats:\n Mean: %f std: %f' % 
        (means[1], stds[1]))

if __name__ == '__main__':
    print("Hello")
    run_env()
