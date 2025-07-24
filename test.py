# test_policy.py
from stable_baselines3 import PPO
from environment import LQREnv  # your custom env
import numpy as np
import matplotlib.pyplot as plt

def test_policy(model_path, n_episodes=50, render=False):
    env = LQREnv()
    model = PPO.load(model_path)

    all_rewards = []
    all_trajectories = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        trajectory = [obs.copy()]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            trajectory.append(obs.copy())

            if render:
                env.render()

        all_rewards.append(episode_reward)
        all_trajectories.append(np.array(trajectory))

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    return all_rewards, all_trajectories


if __name__ == "__main__":
    rewards, trajectories = test_policy("ppo_lqr", n_episodes=5)
    # print("trajectories shape:", trajectories.shape)

    # Example: plot state trajectories of first episode
    traj = trajectories[0]
    plt.plot(traj[:, 0], label="x1 (position)")
    plt.plot(traj[:, 1], label="x2 (velocity)")
    print("State trajectory shape:", traj.shape)
    plt.title("State Trajectory of Episode 1")
    plt.xlabel("Timestep")
    plt.ylabel("State Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ##------------ Plot learned policy u = π(x) ------------

    # # Load model
    # model = PPO.load("ppo_lqr")

    # # Dummy env for normalization, if needed
    # env = LQREnv()

    # # Define the range for each state dimension
    # x1_range = np.linspace(-5, 5, 50)
    # x2_range = np.linspace(-5, 5, 50)

    # # Meshgrid to evaluate the policy on a 2D grid
    # X1, X2 = np.meshgrid(x1_range, x2_range)
    # U = np.zeros_like(X1)

    # # Flatten meshgrid to evaluate efficiently
    # grid_points = np.vstack([X1.ravel(), X2.ravel()]).T

    # for i, x in enumerate(grid_points):
    #     # Make sure obs is np.float32
    #     action, _ = model.predict(x.astype(np.float32), deterministic=True)
    #     U.ravel()[i] = action  # assumes action is scalar

    # plt.figure(figsize=(8, 6))
    # contour = plt.contourf(X1, X2, U, levels=50, cmap="coolwarm")
    # plt.colorbar(contour, label="Action u = π(x)")
    # plt.xlabel("State x₁")
    # plt.ylabel("State x₂")
    # plt.title("Learned Policy u = π(x) by PPO")
    # plt.grid(True)
    # plt.show()





