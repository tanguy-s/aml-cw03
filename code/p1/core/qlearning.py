import time

import numpy as np
import tensorflow as tf

from core.utils import evaluate, reward_value

GAMMA = 0.99
NUM_EPOCHS = 201
LOG_EPOCHS = 100
EVAL_EPOCHS = 3
TARGET_UPDATE = 5

def do_batch_qlearning(env, history_buffer, model, learning_rate):

    tf.reset_default_graph()

    # Create placeholders
    states_pl = tf.placeholder(tf.float32, shape=(None, 4))
    actions_pl= tf.placeholder(tf.int32, shape=(None))
    targets_pl = tf.placeholder(tf.float32, shape=(None))

    # Value function approximator network
    q_output = model.graph(states_pl)

    # Compute Q from current q_output and one hot actions
    Q = tf.reduce_sum(
            tf.multiply(q_output, 
                tf.one_hot(actions_pl, 2, dtype=tf.float32)
            ), axis=1)

    # Loss operation 
    loss_op = tf.reduce_mean(tf.square(targets_pl - Q) / 2)

    # Optimizer Op
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Training Op
    train_op = optimizer.minimize(loss_op)

    # Prediction Op
    prediction = tf.argmax(q_output, 1)

    # Model Saver
    saver = tf.train.Saver()

    # init all variables
    init_op = tf.global_variables_initializer()

    # Start Session
    with tf.Session() as sess:
        sess.run(init_op)

        start_time = time.time()

        # Save results
        losses = list()
        means = list()

        for epoch in range(NUM_EPOCHS):
            loss = 0
            for batch in range(history_buffer.num_batch):

                prev_states, next_states, actions, rewards = history_buffer.next_batch()

                q_out = sess.run(q_output, 
                    feed_dict={
                        states_pl: next_states})

                q_out_max = np.amax(q_out, axis=1)

                q_target = rewards + (1 + np.transpose(rewards)) * GAMMA * q_out_max

                # Run training Op
                l, _ = sess.run([loss_op, train_op], 
                    feed_dict={
                        states_pl: prev_states,
                        actions_pl: actions,
                        targets_pl: q_target
                    })

                loss += l / history_buffer.num_batch

            # Save loss
            losses.append(loss)

            if epoch % LOG_EPOCHS == 0:
                print('')
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                print("Epochs {} to {} done, took {:2f}s".format(max(1, epoch + 1 - LOG_EPOCHS),epoch+1,time.time()-start_time))
                print("Training loss: {:.4f}".format(loss))
                start_time = time.time()

            if epoch % EVAL_EPOCHS == 0:
                silent = (epoch % LOG_EPOCHS != 0)
                cur_means, cur_stds = evaluate(env, sess, prediction, 
                                        states_pl, 20, 300, GAMMA, silent)

                # Save means
                means.append(cur_means)


    # Return Q-learning Experience results
    return losses, means





def do_online_qlearning(env, 
                        model, 
                        learning_rate, 
                        epsilon_s,
                        num_episodes, 
                        len_episodes,
                        target_model=None, 
                        replay_buffer=None,
                        dpaths=None):

    tf.reset_default_graph()

    # Create placeholders
    states_pl = tf.placeholder(tf.float32, shape=(None, 4), name='states')
    actions_pl= tf.placeholder(tf.int32, shape=(None,), name='actions')
    targets_pl = tf.placeholder(tf.float32, shape=(None,), name='targets')

    # Value function approximator network
    q_output = model.graph(states_pl)

    # Build target network
    if target_model:
        q_target = target_model.graph(states_pl)

        trainvars = tf.trainable_variables()
        ntrainvars = len(trainvars)
        target_net_vars = []
        for idx,var in enumerate(trainvars[0:ntrainvars//2]):
            target_net_vars.append(trainvars[idx+ntrainvars//2].assign(var.value()))

    # Compute Q from current q_output and one hot actions
    Q = tf.reduce_sum(
            tf.multiply(q_output, 
                tf.one_hot(actions_pl, 2, dtype=tf.float32)
            ), axis=1)

    # Loss operation 
    loss_op = tf.reduce_mean(tf.square(targets_pl - Q) / 2)

    # Optimizer Op
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Training Op
    train_op = optimizer.minimize(loss_op)

    # Prediction Op
    prediction = tf.argmax(q_output, 1)

    # Model Saver
    saver = tf.train.Saver()

    # init all variables
    init_op = tf.global_variables_initializer()

    # Start Session
    with tf.Session() as sess:
        sess.run(init_op)

        start_time = time.time()

        # Save results
        losses = list()
        means = list()

        #init epsilon
        epsilon = epsilon_s['start']

        for i_episode in range(num_episodes):
            epi_loss, loss = 0, 0

            # Reset environement variables
            state = env.reset()
            done = False
            #retval = 0 # Value function

            for t in range(len_episodes):

                # Epsilon greedy policy
                if epsilon > np.random.rand(1):
                    # Exploration
                    # Use uniformly sampled action from env
                    action = np.array(env.action_space.sample()).reshape([-1])
                else:
                    # Exploitation 
                    # Use model predicted action 
                    actions = sess.run(prediction, feed_dict={
                        states_pl: state.reshape([-1,4]).astype('float32')
                    })
                    action = actions[0]

                # Run next step with action
                next_state, _, done, info = env.step(action)
                reward = reward_value(done)
                #retval += pow(gamma, t)*reward

                if replay_buffer:
                    state = state.reshape([-1,4]).astype('float32')
                    action = action.reshape([-1])
                    next_state = next_state.reshape([-1,4]).astype('float32')

                    # Use experience replay buffer to fill in buffer
                    replay_buffer.add([state, action, reward, next_state])

                    #print(next_state.shape)

                    # If replay buffer is ready do Online learning
                    if replay_buffer.ready:
                        # Train model on replay buffer
                        transitions = replay_buffer.get_rand_transitions()

                        b_states = list()
                        b_actions = list()
                        b_q_target = list()

                        for transition in transitions:

                            #print(transition[-1], type(transition[-1]))
                            # transition: (Si, Ai, Ri+1, Si+1)
                            if target_model:
                                q_out = sess.run(q_target, feed_dict={
                                        states_pl: transition[-1] # Next state
                                    })
                            else:
                                q_out = sess.run(q_output, feed_dict={
                                        states_pl: transition[-1] # Next state
                                    })

                            q_out_max = np.amax(q_out, axis=1)
                            q_target = transition[2] + (1 + np.transpose(transition[2])) * GAMMA * q_out_max

                            # Build batch of experience
                            b_states.append(transition[0])
                            b_actions.append(transition[1])
                            b_q_target.append(q_target)

                        b_states = np.stack(b_states, axis=0)
                        b_actions = np.stack(b_actions, axis=0)
                        b_q_target = np.stack(b_q_target, axis=0)
                            

                        # Run training Op on batch of replay experience
                        loss, _ = sess.run([loss_op, train_op], 
                            feed_dict={
                                states_pl: b_states.reshape([-1,4]),
                                actions_pl: b_actions.reshape([-1]),
                                targets_pl: b_q_target.reshape([-1]).astype('float32')
                            })


                else:
                    q_out = sess.run(q_output, feed_dict={
                                        states_pl: next_state.reshape([-1,4])
                                    })

                    q_out_max = np.amax(q_out, axis=1)
                    q_target = reward + (1 + np.transpose(reward)) * GAMMA * q_out_max

                    # Run training Op
                    loss, _ = sess.run([loss_op, train_op], 
                        feed_dict={
                            states_pl: state.reshape([-1,4]),
                            actions_pl: actions,
                            targets_pl: np.array([q_target]).reshape(-1)
                        })

                epi_loss += loss
                state = next_state
                # break loop on end of episode
                if done:
                    break

            # Update epsilon greedy policy
            if epsilon > epsilon_s['end']:
                epsilon -= (epsilon_s['start'] - epsilon_s['end']) / epsilon_s['decay']

            # Copy variables if target network
            if target_model and (i_episode + 1) % TARGET_UPDATE == 0:
                for var in target_net_vars:
                    sess.run(var)

            # Save loss
            losses.append(epi_loss)

            if i_episode % LOG_EPOCHS == 0:
                print('')
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                print("Epochs {} to {} done, took {:2f}s".format(max(1, i_episode + 1 - LOG_EPOCHS),i_episode+1,time.time()-start_time))
                print("Training loss: {:.4f}".format(epi_loss))
                start_time = time.time()

            if i_episode % EVAL_EPOCHS == 0:
                silent = (i_episode % LOG_EPOCHS != 0)
                cur_means, cur_stds = evaluate(env, sess, prediction, 
                                        states_pl, 20, 300, GAMMA, silent)

                # Save means
                means.append(cur_means)

    # Save models
    if dpaths is not None:
        saver.save(sess, dpaths)

    # Return Q-learning Experience results
    return losses, means





