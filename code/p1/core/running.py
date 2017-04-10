import os
import time
import pickle

import numpy as np
import tensorflow as tf

from core.qlearning import (
    do_batch_qlearning,
    do_online_qlearning
)


def run_multiple_trials_batch(env, 
                                replay_buffer, 
                                model, 
                                learning_rates, 
                                num_runs, 
                                dpaths):
    
    base_path = dpaths[0]
    losses = None
    losses_file = os.path.join(dpaths[0], 'losses.csv')
    losses_pkl = os.path.join(dpaths[0], 'losses.pkl')

    results = None
    results_file = os.path.join(dpaths[0], 'results.csv')
    results_pkl = os.path.join(dpaths[0], 'results.pkl')

    # Loop over learning rates
    i = 1
    for lr in learning_rates:
        print('\n')
        print('###################################')
        print('#### Learning rate %f' % lr)
        print('###################################')
        r_losses = list()
        r_means = list()

        dpaths[0] = os.path.join(base_path, str(i))
        dpaths[1] = os.path.join(dpaths[0], dpaths[2])

        cur_losses_file = os.path.join(dpaths[0], 'losses.csv')
        cur_results_file = os.path.join(dpaths[0], 'results.csv')

        if not os.path.exists(dpaths[0]):
            os.mkdir(dpaths[0])

        i += 1

        # Average results over multiple runs
        for r in range(num_runs):
            print('\n## Run %d/%d' % (r+1, num_runs))

            cur_losses, cur_means = do_batch_qlearning(env, replay_buffer, model, lr, dpaths)

            # Save run specific results
            r_losses.append(cur_losses)
            r_means.append(cur_means)

        # Save LR specific results
        r_losses_mean = np.mean(np.array(r_losses), axis=0).reshape([-1, 1])
        if losses is None:
            losses = r_losses_mean
        else:
            losses = np.concatenate([losses, r_losses_mean], axis=1)  

        np.savetxt(cur_losses_file, losses, delimiter=',')

        r_results = np.concatenate([
            np.mean(np.array(r_means), axis=0), 
            np.std(np.array(r_means), axis=0)], axis=1)[:, [0,2,1,3]]

        #r_results = np.mean(np.array(r_means), axis=0) 
        if results is None:
            results = r_results
        else:
            results = np.concatenate([results, r_results], axis=1)

        np.savetxt(cur_results_file, results, delimiter=',')


    with open(losses_pkl, 'wb') as f:
        pickle.dump(losses, f)

    with open(results_pkl, 'wb') as f:
        pickle.dump(results, f)

    # Handle file saving
    np.savetxt(losses_file, losses, delimiter=',')
    np.savetxt(results_file, results, delimiter=',')


def run_multiple_trials_online(env, main_model, num_runs, dpaths):
    
    losses = None
    losses_file = os.path.join(
        dpaths[0], 'losses.csv')
    losses_pkl = os.path.join(dpaths[0], 'losses.pkl')

    results = None
    results_file = os.path.join(
            dpaths[0], 'results.csv')
    results_pkl = os.path.join(dpaths[0], 'results.pkl')

    r_losses = list()
    r_means = list()

    # Average results over multiple runs
    for r in range(num_runs):
        print('\n## Run %d/%d' % (r+1, num_runs))

        cur_losses, cur_means = do_online_qlearning(env, 
                                    model=main_model.model, 
                                    learning_rate=main_model.learning_rate,
                                    epsilon_s=main_model.epsilon_s, 
                                    num_episodes=main_model.num_episodes,
                                    len_episodes=main_model.len_episodes,
                                    target_model=main_model.target_model,
                                    replay_buffer=main_model.replay_buffer,
                                    dpaths=dpaths)

        # Save run specific results
        r_losses.append(cur_losses)
        r_means.append(cur_means)

    # Save LR specific results
    r_losses_mean = np.mean(np.array(r_losses), axis=0).reshape([-1, 1])
    if losses is None:
        losses = r_losses_mean
    else:
        losses = np.concatenate([losses, r_losses_mean], axis=1)   

    r_results = np.concatenate([
        np.mean(np.array(r_means), axis=0), 
        np.std(np.array(r_means), axis=0)], axis=1)[:, [0,2,1,3]]

    #r_results = np.mean(np.array(r_means), axis=0) 
    if results is None:
        results = r_results
    else:
        results = np.concatenate([results, r_results], axis=1)

    with open(losses_pkl, 'wb') as f:
        pickle.dump(losses, f)

    with open(results_pkl, 'wb') as f:
        pickle.dump(results, f)

    # Handle file saving
    np.savetxt(losses_file, losses, delimiter=',')
    np.savetxt(results_file, results, delimiter=',')


    



    





