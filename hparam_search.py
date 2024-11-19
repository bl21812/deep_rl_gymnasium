import optuna
import gymnasium as gym
import highway_env

from utils import eval, create_model_optuna

# NOTE: DQN does not work with racetrack, and SAC only works with racetrack
envs = ['highway-fast-v0', 'intersection-v0', 'racetrack-v0']
agents = ['A2C', 'PPO', 'DQN', 'SAC']

NUM_TRIALS = 20
ENV = envs[0]
AGENT = agents[0]

TRAIN_TIMESTEPS = int(1e4)
    
# create environment
env = gym.make(ENV, render_mode='rgb_array')
    
# create study using the TPESampler.
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.HyperbandPruner()
)

# optuna run trial function (including eval)
def run_optuna_trial(trial):

    # initialize
    env.reset()
    model = create_model_optuna(model_type=AGENT, env=env, trial=trial, env_type=ENV)

    # train
    model.learn(TRAIN_TIMESTEPS)

    # eval
    return eval(model, env, trial=trial, model_type=AGENT, env_type=ENV)

# run optuna study to for highest avg reward hparams
study.optimize(run_optuna_trial, n_trials=NUM_TRIALS)

# Print the params of the most optimal study
print(f"Optimal Values Found in {NUM_TRIALS} trials:")
print("-------------------------------------------------")
for param, optimum_val in study.best_trial.params.items():
    print(f"{param} : {optimum_val}")
