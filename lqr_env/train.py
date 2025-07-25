from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environment import LQREnv
from stable_baselines3.common.monitor import Monitor
import os
log_dir = "./ppo_lqr_logs/"
os.makedirs(log_dir, exist_ok=True)

# Define the environment creation function
def make_env():
    def _init():
        env = LQREnv()
        return Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    return _init

# Create vectorized environment
vec_env = make_vec_env(make_env(), n_envs=1)

# Define policy architecture
policy_kwargs = dict(
    net_arch=[dict(pi=[32, 32], vf=[32, 32])]  # pi = policy net, vf = value net
)

model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=200000)
model.save("ppo_lqr")

# Load and test
model = PPO.load("ppo_lqr")
obs = vec_env.reset()

for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.envs[0].render()  # render one env


import pandas as pd
import matplotlib.pyplot as plt

monitor_file = os.path.join(log_dir, "monitor.csv")
data = pd.read_csv(monitor_file, skiprows=1)  # skip SB3 header
plt.plot(data["l"].cumsum(), data["r"])  # cumulative timesteps vs reward
plt.xlabel("Timestep")
plt.ylabel("Episode Reward")
plt.title("Reward per Episode")
plt.grid(True)
plt.show()
