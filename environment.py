import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LQREnv(gym.Env):
    def __init__(self):
        super(LQREnv, self).__init__()

        self.max_steps = 100
        self.current_step = 0

        # System dynamics
        self.A = np.eye(2)
        self.B = np.eye(2)
        # self.A = np.array([[1.0, 1.0], [0, 1.0]])
        # self.B = np.array([[1.0], [1.0]])

        # Cost matrices
        self.Q = np.eye(2)
        self.R = np.eye(2) * 0.1

        # Observation and action spaces
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(low=-1.0, high=1.0, size=(2,))
        self.current_step = 0
        return self.state.astype(np.float32), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        x = self.state
        u = action

        # LQR dynamics
        u = 5 * np.atleast_1d(action).astype(np.float32)  # shape (1,)
        x_next = self.A @ x + self.B @ u  # B @ u is shape (2,)


        # LQR cost (negative reward)
        cost = x.T @ self.Q @ x + u.T @ self.R @ u
        reward = -cost.item() / 1000

        self.state = x_next
        done = False  # You can terminate after fixed steps or based on state norm

        self.current_step += 1

        # Terminate if too many steps or state too large
        done = self.current_step >= self.max_steps or np.linalg.norm(self.state) > 100000.0

        return x_next.astype(np.float32), reward, done, False, {}

    def render(self, mode="human"):
        print(f"State: {self.state}")
