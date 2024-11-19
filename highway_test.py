# NOTE: docs here https://highway-env.farama.org/quickstart/
# NOTE: env setup works as per the 'open in colab' notebooks from the link above

import gymnasium
import highway_env
from stable_baselines3 import A2C, PPO, DQN

envs = ['highway-fast-v0', 'intersection-v0', 'racetrack-v0']
agents = ['A2C', 'PPO', 'DQN']

# ----- SAMPLE CODE FROM HIGHWAY-ENV DOCS -----
env = gymnasium.make("highway-fast-v0", render_mode='rgb_array')
'''model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,
              tensorboard_log="highway_dqn/")
model.learn(int(2e4))
model.save("highway_dqn/model")'''

# Load and test saved model
# This plays a nice little animation
model = DQN.load("highway_dqn/model")

runs = 0
crashes = 0
reward_history = []

while runs < 10:
  done = truncated = False
  obs, info = env.reset()

  # RUNS THROUGH A SCENARIO - sum reward to get return or smth (avg? weighted avg?)
  reward_history.append([])
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    '''print(obs)
    print(reward)
    print(done)
    print(truncated)
    print(info)
    input()'''

    reward_history[-1].append(reward) 

    if info['crashed']:
        crashes += 1

    # env.render()

  runs += 1

print(runs)
print(crashes)

for i in range(runs):
  print(reward_history[i])
