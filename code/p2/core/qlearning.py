import sys
import time

import numpy as np
import tensorflow as tf

from core.utils import evaluate, reward_value, do_obs_processing, reward_clip

GAMMA = 0.99
TRAINING_STEPS = 500000
LOG_STEPS = 5000
LOSS_STEPS = 1000
EVAL_STEPS = 50000
SAVE_STEPS = 50000
TARGET_UPDATE = 5000
FRAME_WIDTH = 84
FRAME_HEIGHT = 84
FRAME_BUFFER_SIZE = 4


def do_online_qlearning(env, 
                        test_env,    
                        model, 
                        learning_rate, 
                        epsilon_s,
                        gpu_device,
                        target_model=None, 
                        replay_buffer=None,
                        dpaths=None):

    tf.reset_default_graph()

    with tf.device(gpu_device):

        # Create placeholders
        states_pl = tf.placeholder(tf.float32, 
            shape=(None, FRAME_WIDTH, FRAME_HEIGHT, 4), name='states')
        actions_pl= tf.placeholder(tf.int32, shape=(None), name='actions')
        targets_pl = tf.placeholder(tf.float32, shape=(None), name='targets')

        # Value function approximator network
        q_output = model.graph(states_pl)

        # Build target network
        q_target_net = target_model.graph(states_pl)

        trainvars = tf.trainable_variables()
        ntrainvars = len(trainvars)
        target_net_vars = []
        for idx,var in enumerate(trainvars[0:ntrainvars//2]):
            target_net_vars.append(trainvars[idx+ntrainvars//2].assign(var.value()))

        # Compute Q from current q_output and one hot actions
        Q = tf.reduce_sum(
                tf.multiply(q_output, 
                    tf.one_hot(actions_pl, env.action_space.n, dtype=tf.float32)
                ), axis=1)

        # Loss operation 
        loss_op = tf.reduce_mean(tf.square(targets_pl - Q) / 2)

        # Optimizer Op
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

        # Training Op
        train_op = optimizer.minimize(loss_op)

        # Prediction Op
        prediction = tf.argmax(q_output, 1)

    # Model Saver
    saver = tf.train.Saver()

    # init all variables
    init_op = tf.global_variables_initializer()

    # Limit memory usage for multiple training at same time
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.33

    # Start Session
    with tf.Session(config=config) as sess:
        sess.run(init_op)

        start_time = time.time()

        # Load env and get observations
        observation = env.reset()

        # Observation Buffer
        observation_buffer = list()

        # Save results
        losses = list()
        means = list()

        #init epsilon
        epsilon = epsilon_s['start']

        for step in range(TRAINING_STEPS):
            loss = 0
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

                # Epsilon greedy policy
                if epsilon > np.random.rand(1):
                    # Exploration
                    # Use uniformly sampled action from env
                    action = np.array(env.action_space.sample(), dtype=np.int32).reshape((-1))
                else:
                    # Exploitation 
                    # Use model predicted action 
                    action = sess.run(prediction, feed_dict={
                        states_pl: state.reshape(
                            [-1, FRAME_WIDTH, FRAME_HEIGHT, FRAME_BUFFER_SIZE]).astype('float32')
                    })

                # action for next observation
                observation, reward, done, info  = env.step(action[0])

                # Clip reward
                r = reward_clip(reward)

                # Update observation buffer
                observation_buffer.append(
                    do_obs_processing(observation, FRAME_WIDTH, FRAME_HEIGHT))
                observation_buffer[0:1] = []

                next_state = np.stack(observation_buffer, axis=-1)
                action = action.reshape([-1]).astype('int32')

                # Add transition to replay buffer
                replay_buffer.add((state, action, r, next_state, done))

                # If replay buffer is ready to be sampled
                if replay_buffer.ready:
                    # Train model on replay buffer
                    transitions = replay_buffer.get_rand_transitions()

                    states_batch = np.stack([transitions[i][0] for i in range(len(transitions))], axis=0).astype('float32')
                    actions_batch = np.stack([transitions[i][1] for i in range(len(transitions))], axis=0).reshape((-1))
                    r_batch = np.stack([transitions[i][2] for i in range(len(transitions))], axis=0).reshape((-1))
                    next_state_batch = np.stack([transitions[i][3] for i in range(len(transitions))], axis=0).astype('float32')
                    terminal_state_batch = np.stack([transitions[i][4] for i in range(len(transitions))], axis=0)

                    # Run training on batch
                    q_out = sess.run(q_target_net, feed_dict={
                            states_pl: next_state_batch
                        })
                    q_out_max = np.amax(q_out, axis=1)
                    q_target = r_batch + GAMMA * (1 - terminal_state_batch) * q_out_max

                    # Run training Op on batch of replay experience
                    loss, _ = sess.run([loss_op, train_op], 
                        feed_dict={
                            states_pl: states_batch,
                            actions_pl: actions_batch,
                            targets_pl: q_target.astype('float32')
                        })
    
            if done:
                observation = env.reset()

            # Update epsilon greedy policy
            if epsilon > epsilon_s['end']:
                epsilon -= (epsilon_s['start'] - epsilon_s['end']) / epsilon_s['decay']

            # Copy variables target network
            if (step + 1) % TARGET_UPDATE == 0:
                for var in target_net_vars:
                    sess.run(var)

            if step % LOSS_STEPS == 0:
                # Save loss
                losses.append(loss)

            if step % LOG_STEPS == 0:
                print('')
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                print("Steps {} to {} done, took {:.2f}s".format(max(0,step+1-LOG_STEPS),step,time.time()-start_time))
                print(" Train loss: {:.4f}".format(loss))
                start_time = time.time()

                # Force flush for nohup
                sys.stdout.flush()

            if step % EVAL_STEPS == 0:
                silent = (step % LOG_STEPS != 0)
                cur_means, cur_stds = evaluate(test_env, sess, prediction, 
                                        states_pl, 5, GAMMA, silent)

                # Save means
                means.append(cur_means)

            # Save models
            if dpaths is not None and step % SAVE_STEPS == 0:
                saver.save(sess, dpaths, global_step=step)
             
        # Save models
        if dpaths is not None:
            saver.save(sess, dpaths)

    # Return Q-learning Experience results
    return losses, means