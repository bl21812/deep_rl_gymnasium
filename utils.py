# Helper functions

from stable_baselines3 import A2C, PPO, DQN

def eval(model, env, runs=10):
    '''
    Evaluate a model - run inference and return average reward
    
    :param model: Stable baselines model
    :param env: Gymnasium environment
    :param runs: Runs to be performed (to average reward over)
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

        runs += 1
        avg_rewards.append(sum(reward_history)[-1] / len(reward_history[-1]))

    # calculate average reward across all runs
    avg_run_reward = sum(avg_rewards) / len(avg_rewards)

    return avg_run_reward


def create_model(model_type, env, hparams):
    '''
    Create a Stable baselines model
    '''
    
    if model_type == 'DQN':
        model = DQN('MlpPolicy', env,
            policy_kwargs=dict(net_arch=[256, 256]),
            gradient_steps=1,
            verbose=1,
            tensorboard_log="highway_dqn/",
            **hparams)

    elif model_type == 'PPO':
        model = None

    elif model_type == 'A2C':
        model = None

    return model


# TODO: ADD HPARAM RANGES HERE !!
def create_model_optuna(model_type, env, trial):
    '''
    Create a Stable baselines model with hparams suggested by Optuna
    '''

    if model_type == 'DQN':
        hparams = {
            'learning_rate': trial.suggest_float(),
            'buffer_size': trial.suggest_int(),
            'learning_starts': trial.suggest_int(),
            'batch_size': trial.suggest_int(),
            'gamma': trial.suggest_float(),
            'train_freq': trial.suggest_int(),
            'target_update_interval': trial.suggest_int(),
        }
        model = create_model(model_type, env, hparams)

    elif model_type == 'PPO':
        model = None

    elif model_type == 'A2C':
        model = None

    return model
