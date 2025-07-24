# rl_examples

# PPO on LQR Example

This repository contains a minimal example of solving a linear-quadratic regulation (LQR) problem using reinforcement learning (PPO from Stable-Baselines3).

---

## 📦 Environment Setup

We recommend using **Conda** to manage dependencies.

### 1. Create a Conda virtual environment

```bash
conda create -n rl_examples python=3.9
conda activate rl_examples
pip install numpy matplotlib gymnasium
pip install stable-baselines3[extra]

## 🧪 Files

| File             | Description                                     |
| ---------------- | ----------------------------------------------- |
| `environment.py` | Defines the custom LQR Gymnasium environment    |
| `train.py`       | Trains the PPO agent on the LQR environment     |
| `test.py`        | Evaluates the trained agent                     |
| `plot_policy.py` | Visualizes the policy function over state space |
