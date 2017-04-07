import sys
import time

import numpy as np
import tensorflow as tf

from core.utils import evaluate, reward_value

GAMMA = 0.99
NUM_EPOCHS = 201
LOG_EPOCHS = 100
EVAL_EPOCHS = 4
TARGET_UPDATE = 5


def do_batch_qlearning(env, history_buffer, model, learning_rate, dpaths=None):

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

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Start Session
    with tf.Session(config=config) as sess:
        sess.run(init_op)

        start_time = time.time()

        # Save results
        losses = list()
        means = list()

        for epoch in range(NUM_EPOCHS):
            loss = 0
            for batch in range(history_buffer.num_batch):

                prev_states, next_states, actions, rewards, term_state = history_buffer.next_batch()

                q_out = sess.run(q_output, 
                    feed_dict={
                        states_pl: next_states})

                q_out_max = np.amax(q_out, axis=1)

                q_target = rewards + (1 - np.transpose(term_state)) * GAMMA * q_out_max

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
                print('\n', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
                print('Epoch %d -> %d Done (%.3fs) ... ' % 
                    (max(1, epoch + 1 - LOG_EPOCHS), epoch+1, time.time() - start_time))
                print('- Training loss: %.4f' % loss)
                start_time = time.time()

            if epoch % EVAL_EPOCHS == 0:
                silent = (epoch % LOG_EPOCHS != 0)
                cur_means, cur_stds = evaluate(env, sess, prediction, 
                                        states_pl, 20, 300, GAMMA, silent)

                # Save means
                means.append(cur_means)

            # Force flush for nohup
            sys.stdout.flush()

        # Save models
        if dpaths is not None:
            print('## Saved model !')
            saver.save(sess, dpaths[1])

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
        q_target_net = target_model.graph(states_pl)

        tf_train = tf.trainable_variables()
        num_tf_train = len(tf_train)
        target_net_vars = []
        for i, var in enumerate(tf_train[0:num_tf_train // 2]):
            target_net_vars.append(tf_train[i + num_tf_train // 2].assign(var.value()))

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
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
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
                    actions = np.array(env.action_space.sample()).reshape([-1])
                else:
                    # Exploitation 
                    # Use model predicted action 
                    actions = sess.run(prediction, feed_dict={
                        states_pl: state.reshape([-1,4]).astype('float32')
                    })

                # Run next step with action
                next_state, _, done, info = env.step(actions[0])
                reward = reward_value(done)
                #retval += pow(gamma, t)*reward

                if replay_buffer:
                    state = state.reshape([-1,4]).astype('float32')
                    actions = actions.reshape([-1])
                    next_state = next_state.reshape([-1,4]).astype('float32')

                    # Use experience replay buffer to fill in buffer
                    replay_buffer.add((state, actions, reward, next_state, done))

                    #print(next_state.shape)

                    # If replay buffer is ready do Online learning
                    if replay_buffer.ready:
                        # Train model on replay buffer
                        b_states, b_actions, b_reward, b_next_state, b_term_state = replay_buffer.next_transitions()

                        if target_model:
                            q_out = sess.run(q_target_net, feed_dict={
                                    states_pl: b_next_state
                                })
                        else:
                            q_out = sess.run(q_output, feed_dict={
                                    states_pl: b_next_state
                                })

                        q_out_max = np.amax(q_out, axis=1)
                        b_q_target = b_reward + (1 - np.transpose(b_term_state)) * GAMMA * q_out_max

                        # Run training Op on batch of replay experience
                        loss, _ = sess.run([loss_op, train_op], 
                            feed_dict={
                                states_pl: b_states,
                                actions_pl: b_actions,
                                targets_pl: b_q_target
                            })


                else:
                    q_out = sess.run(q_output, feed_dict={
                                        states_pl: next_state.reshape([-1,4])
                                    })

                    q_out_max = np.amax(q_out, axis=1)
                    q_target = reward + (1 - np.transpose(done)) * GAMMA * q_out_max

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
                print('\n', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
                print('Episodes %d -> %d Done (%.3fs) ... ' % 
                    (max(1, i_episode + 1 - LOG_EPOCHS), i_episode+1, time.time() - start_time))
                print('- Training loss: %.4f' % epi_loss)
                start_time = time.time()

            if i_episode % EVAL_EPOCHS == 0:
                silent = (i_episode % LOG_EPOCHS != 0)
                cur_means, cur_stds = evaluate(env, sess, prediction, 
                                        states_pl, 20, 300, GAMMA, silent)

                # Save means
                means.append(cur_means)

            # Force flush for nohup
            sys.stdout.flush()

        # Save models
        if dpaths is not None:
            print('## Saved model !')
            saver.save(sess, dpaths[1])

    # Return Q-learning Experience results
    return losses, means


def do_online_double_qlearning(env, 
                                model_1, 
                                model_2,
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
    q_output_1 = model_1.graph(states_pl)
    q_output_2 = model_2.graph(states_pl)

    # Compute Q from current q_output and one hot actions
    Q_1 = tf.reduce_sum(
            tf.multiply(q_output_1, 
                tf.one_hot(actions_pl, 2, dtype=tf.float32)
            ), axis=1)
    Q_2 = tf.reduce_sum(
            tf.multiply(q_output_2, 
                tf.one_hot(actions_pl, 2, dtype=tf.float32)
            ), axis=1)

    # Loss operation 
    loss_op_1 = tf.reduce_mean(tf.square(targets_pl - Q_1) / 2)
    loss_op_2 = tf.reduce_mean(tf.square(targets_pl - Q_2) / 2)

    # Optimizer Op
    optimizer_1 = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer_2 = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Training Op
    train_op_1 = optimizer_1.minimize(loss_op_1)
    train_op_2 = optimizer_2.minimize(loss_op_2)

    # Prediction Op
    prediction_1 = tf.argmax(q_output_1, 1)
    prediction_2 = tf.argmax(q_output_2, 1)

    # Model Saver
    saver = tf.train.Saver()

    # init all variables
    init_op = tf.global_variables_initializer()

    # Start Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
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
                    actions = np.array(env.action_space.sample()).reshape([-1])
                else:
                    # Exploitation 
                    # Use model predicted action 
                    q_out_1 = sess.run(q_output_1, feed_dict={
                        states_pl: state.reshape([-1,4]).astype('float32')
                    }) 
                    q_out_2 = sess.run(q_output_2, feed_dict={
                        states_pl: state.reshape([-1,4]).astype('float32')
                    }) 
                    actions = np.argmax(q_out_1 + q_out_2, axis=1).astype('int32')

                # Run next step with action
                next_state, _, done, info = env.step(actions[0])
                reward = reward_value(done)
                #retval += pow(gamma, t)*reward

                if replay_buffer:
                    state = state.reshape([-1,4]).astype('float32')
                    actions = actions.reshape([-1])
                    next_state = next_state.reshape([-1,4]).astype('float32')

                    # Use experience replay buffer to fill in buffer
                    replay_buffer.add((state, actions, reward, next_state, done))

                    #print(next_state.shape)

                    # If replay buffer is ready do Online learning
                    if replay_buffer.ready:
                        # Train model on replay buffer
                        b_states, b_actions, b_reward, b_next_state, b_term_state = replay_buffer.next_transitions()

                        if np.random.rand(1) > 0.50:

                            actions_pred = sess.run(prediction_1, feed_dict={
                                        states_pl: b_next_state
                                    })
                            b_q_out = sess.run(q_output_1, feed_dict={
                                        states_pl: b_next_state
                                    }) 
                            q_out_max = b_q_out[np.arange(np.shape(b_q_out)[0]), actions_pred]
                            b_q_target = b_reward + (1 - np.transpose(b_term_state)) * GAMMA * q_out_max

                            # Run training Op on batch of replay experience
                            loss, _ = sess.run([loss_op_1, train_op_1], 
                                feed_dict={
                                    states_pl: b_states,
                                    actions_pl: b_actions,
                                    targets_pl: b_q_target
                                })

                        else:
                            actions_pred = sess.run(prediction_2, feed_dict={
                                        states_pl: b_next_state
                                    })
                            b_q_out = sess.run(q_output_2, feed_dict={
                                        states_pl: b_next_state
                                    }) 
                            q_out_max = b_q_out[np.arange(np.shape(b_q_out)[0]), actions_pred]
                            b_q_target = b_reward + (1 - np.transpose(b_term_state)) * GAMMA * q_out_max

                            # Run training Op on batch of replay experience
                            loss, _ = sess.run([loss_op_2, train_op_2], 
                                feed_dict={
                                    states_pl: b_states,
                                    actions_pl: b_actions,
                                    targets_pl: b_q_target
                                })


                else:
                    q_out = sess.run(q_output, feed_dict={
                                        states_pl: next_state.reshape([-1,4])
                                    })

                    q_out_max = np.amax(q_out, axis=1)
                    q_target = reward + (1 - np.transpose(done)) * GAMMA * q_out_max

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

            # Save loss
            losses.append(epi_loss)

            if i_episode % LOG_EPOCHS == 0:
                print('\n', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
                print('Episodes %d -> %d Done (%.3fs) ... ' % 
                    (max(1, i_episode + 1 - LOG_EPOCHS), i_episode+1, time.time() - start_time))
                print('- Training loss: %.4f' % epi_loss)
                start_time = time.time()

            if i_episode % EVAL_EPOCHS == 0:
                silent = (i_episode % LOG_EPOCHS != 0)
                cur_means, cur_stds = evaluate(env, sess, prediction_1, 
                                        states_pl, 20, 300, GAMMA, silent)

                # Save means
                means.append(cur_means)

            # Force flush for nohup
            sys.stdout.flush()

        # Save models
        if dpaths is not None:
            print('## Saved model !')
            saver.save(sess, dpaths[1])

    # Return Q-learning Experience results
    return losses, means





