import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=10)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=1)
model.save("ppo_cartpole_1")

# del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole_1", env=vec_env)

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")