# Helper functions

import os
import numpy as np

from stable_baselines3 import A2C, PPO, DQN

def eval(model, env, trial=None, runs=10, model_type=None, env_type=None):
    '''
    Evaluate a model - run inference and return average reward
    
    :param model: Stable baselines model
    :param env: Gymnasium environment
    :param trial: Optuna trial object
    :param runs: Runs to be performed (to average reward over)
    :param model_type: String identifying type of model (for save directory)
    :return avg_rewards: Reward averaged across timesteps, then across runs
    '''

    runs_done = 0
    reward_history = []
    avg_rewards = []

    while runs_done < runs:

        done = truncated = False
        obs, info = env.reset()

        # RUNS THROUGH A SCENARIO - sum reward to get return or smth (avg? weighted avg?)
        reward_history.append([])
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            reward_history[-1].append(reward) 

        runs_done += 1
        avg_rewards.append(sum(reward_history[-1]) / len(reward_history[-1]))

    # calculate average reward across all runs
    avg_run_reward = sum(avg_rewards) / len(avg_rewards)

    # if trial is passed - save results
    if trial and model_type and env_type:
        save_dir = os.path.join('val_results', env_type, model_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        arr = make_ragged_array(reward_history)
        np.save(os.path.join(save_dir, f'{trial.number}.npy'), arr)

    return avg_run_reward


def create_model(model_type, env, hparams, env_type=None):
    '''
    Create a Stable baselines model
    '''
    
    if model_type == 'DQN':
        model = DQN('MlpPolicy', env,
            policy_kwargs=dict(net_arch=[256, 256]),
            gradient_steps=1,
            tensorboard_log=f"{env_type}_{model_type}/",
            **hparams)

    elif model_type == 'PPO':
        model = PPO('MlpPolicy', env, 
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            tensorboard_log=f"{env_type}_{model_type}/",
            **hparams
        )

    elif model_type == 'A2C':
        model = None

    return model


# TODO: ADD HPARAM RANGES HERE !!
def create_model_optuna(model_type, env, trial, env_type=None):
    '''
    Create a Stable baselines model with hparams suggested by Optuna
    '''

    if model_type == 'DQN':
        hparams = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-2),
            'buffer_size': trial.suggest_int('buffer_size', 1e3, 1e5),
            'learning_starts': trial.suggest_int('learning_starts', 0, 1000),
            'batch_size': trial.suggest_int('batch_size', 4, 64),
            'gamma': trial.suggest_float('gamma', 0.5, 0.999),
            'train_freq': trial.suggest_int('train_freq', 1, 10),
            'target_update_interval': trial.suggest_int('target_update_interval', 25, 100),
        }
        
    elif model_type == 'PPO':
        batch_size = trial.suggest_int('batch_size', 4, 64)
        hparams = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-2),
            'n_steps': batch_size * trial.suggest_int('n_steps_multiplier', 1, 8),  # keep as multiplier of batch size to avoid truncation
            'n_epochs': trial.suggest_int('n_epochs', 2, 50),
            'batch_size': batch_size,
            'gamma': trial.suggest_float('gamma', 0.5, 0.999),
        }

    elif model_type == 'A2C':
        model = None

    model = create_model(model_type, env, hparams, env_type=env_type)

    return model


def make_ragged_array(lists):

    max_length = max([len(l) for l in lists])
    arr = np.zeros((len(lists), max_length))

    for i in range(len(lists)):
        arr[i] = lists[i] + ([0] * (max_length - len(lists[i])))

    return arr
