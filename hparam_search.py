import yaml

import optuna
import gymnasium
import highway_env
from stable_baselines3 import A2C, PPO, DQN

from utils import eval, create_model_optuna

envs = ['highway-fast-v0', 'intersection-v0', 'racetrack-v0']
agents = ['A2C', 'PPO', 'DQN']

ENV = envs[0]
NUM_TRIALS = 10
AGENT = agents[-1]
TRAIN_TIMESTEPS = int(1e4)

agent_hparams = None
with open("config.yml") as cfg:
    try:
        agent_hparams = yaml.safe_load(cfg)
    except yaml.YAMLError as err:
        raise RuntimeError(err)
    
# create environment
env = gymnasium.make(ENV, render_mode='rgb_array')
    
# create study using the TPESampler.
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.HyperbandPruner()
)

# TODO: SAVE MODEL while running trial (compare to best?)
# optuna run trial function (including eval)
def run_optuna_trial(trial):

    # initialize
    env.reset()
    model = create_model_optuna(model_type=AGENT, env=env, trial=trial)

    # train
    model.learn(TRAIN_TIMESTEPS)

    # eval
    return eval(model, env)

# run optuna study to for highest avg reward hparams
# TODO: OBJ FUNCTION must only take in trial as param (should run the whole thing)
study.optimize(run_optuna_trial, n_trials=NUM_TRIALS)
