import gym
env = gym.make('CartPole-v0')


print(env.action_space)
print(env.observation_space)

print(env.observation_space.high)
print(env.observation_space.low)

NUM_EPISODES = 2000
EPISODE_LENGTH = 300


for i_episode in range(NUM_EPISODES):

    # Reset environement
    observation = env.reset()

    for t in range(EPISODE_LENGTH):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print('Reward is %f' % reward)
            break
