# Helper functions

import os
import numpy as np
import gymnasium as gym

from stable_baselines3 import A2C, PPO, DQN, SAC

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
    
    Note:   - SAC only works with continuous action spaces
            - DQN only works with discrete action spaces
    '''
    
    if model_type == 'SAC' and not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError("SAC only supports continuous action spaces (Box). Use A2C, PPO, or DQN for discrete actions.")

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
            device='cpu',
            **hparams)

    elif model_type == 'A2C':
        model = A2C('MlpPolicy', env, 
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            tensorboard_log=f"{env_type}_{model_type}/",
            device='cpu',
            **hparams)

    elif model_type == 'SAC':
        model = SAC('MlpPolicy', env,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=f"{env_type}_{model_type}/",
            **hparams)

    else:
        raise ValueError(f"model_type {model_type} not in ['DQN', 'PPO', 'A2C', 'SAC']")

    return model


# TODO: ADD HPARAM RANGES HERE !!
def create_model_optuna(model_type, env, trial, env_type=None):
    '''
    Create a Stable baselines model with hparams suggested by Optuna
    '''

    if model_type == 'DQN':
        hparams = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True),
            'buffer_size': trial.suggest_int('buffer_size', 10000, 1000000),
            'learning_starts': trial.suggest_int('learning_starts', 1000, 10000),
            'batch_size': trial.suggest_int('batch_size', 32, 256),
            'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
            'train_freq': trial.suggest_int('train_freq', 1, 10),
            'gradient_steps': trial.suggest_int('gradient_steps', 1, 10),
            'target_update_interval': trial.suggest_int('target_update_interval', 100, 1000),
            'exploration_fraction': trial.suggest_float('exploration_fraction', 0.1, 0.5),
            'exploration_initial_eps': trial.suggest_float('exploration_initial_eps', 0.8, 1.0),
            'exploration_final_eps': trial.suggest_float('exploration_final_eps', 0.01, 0.1),
        }
        
    elif model_type == 'PPO':
        batch_size = trial.suggest_int('batch_size', 32, 256)
        hparams = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True),
            'n_steps': batch_size * trial.suggest_int('n_steps_multiplier', 1, 8),  # keep as multiplier of batch size to avoid truncation
            'n_epochs': trial.suggest_int('n_epochs', 2, 50),
            'batch_size': batch_size,
            'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
            'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
            'vf_coef': trial.suggest_float('vf_coef', 0.1, 0.9),
            'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 0.9),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.999),
            'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.01),
        }
    elif model_type == 'A2C':
        hparams = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True),
            'n_steps': trial.suggest_int('n_steps', 8, 2048),
            'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.999),
            'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.01),
            'vf_coef': trial.suggest_float('vf_coef', 0.1, 0.9),
            'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 0.9),
            'rms_prop_eps': trial.suggest_float('rms_prop_eps', 1e-5, 1e-3, log=True),
            'normalize_advantage': trial.suggest_categorical('normalize_advantage', [True, False])
        }
    elif model_type == 'SAC':
        hparams = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True),
            'buffer_size': trial.suggest_int('buffer_size', 10000, 1000000),
            'learning_starts': trial.suggest_int('learning_starts', 1000, 10000),
            'batch_size': trial.suggest_int('batch_size', 32, 256),
            'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
            'train_freq': trial.suggest_int('train_freq', 1, 10),
            'gradient_steps': trial.suggest_int('gradient_steps', 1, 10),
            'tau': trial.suggest_float('tau', 0.001, 0.05),
            'ent_coef': 'auto',  # Using automatic entropy tuning
            'target_entropy': trial.suggest_float('target_entropy', -10, -1),
            'use_sde': trial.suggest_categorical('use_sde', [True, False]),
            'sde_sample_freq': trial.suggest_int('sde_sample_freq', -1, 10),
        }
    else:
        raise ValueError(f"model_type {model_type} not in ['DQN', 'PPO', 'A2C', 'SAC']")

    model = create_model(model_type, env, hparams, env_type=env_type)

    return model


def make_ragged_array(lists):

    max_length = max([len(l) for l in lists])
    arr = np.zeros((len(lists), max_length))

    for i in range(len(lists)):
        arr[i] = lists[i] + ([0] * (max_length - len(lists[i])))

    return arr
