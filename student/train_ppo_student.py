# student/train_ppo_student.py

from typing import Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from env.overcooked_wrapper import OvercookedGym


class StudentPPO:
    """
    Wrapper around a SB3 PPO agent that can:
    - train on a given layout
    - return the average reward over several test episodes
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        n_steps: int = 2048,
        gamma: float = 0.99,
        verbose: int = 1,
    ):
        # Initialize the model on a default layout ("cramped_room")
        def make_env():
            return OvercookedGym("cramped_room", horizon=400)

        vec_env = DummyVecEnv([make_env])
        vec_env = VecMonitor(vec_env)

        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            verbose=verbose,
        )

    def _make_vec_env(self, layout_name: str):
        # Create a new vectorized environment for a specific layout
        def make_env():
            return OvercookedGym(layout_name, horizon=400)

        vec_env = DummyVecEnv([make_env])
        vec_env = VecMonitor(vec_env)
        return vec_env

    def train_on_layout(
        self,
        layout_name: str,
        total_timesteps: int = 20_000,
        eval_episodes: int = 5,
    ) -> float:
        """
        Train the student on a specific layout, then evaluate the average reward.
        - total_timesteps: additional training steps (the global step counter is not reset).
        """
        # New environment for this layout
        vec_env = self._make_vec_env(layout_name)
        self.model.set_env(vec_env)

        # Continue training (reset_num_timesteps=False to keep the internal timestep counter)
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)

        # Then evaluate on several episodes
        avg_return = self._evaluate_layout(layout_name, eval_episodes)
        return avg_return

    def _evaluate_layout(self, layout_name: str, n_episodes: int = 5) -> float:
        # Create a non-vectorized env for evaluation on the given layout
        env = OvercookedGym(layout_name, horizon=400)
        returns = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            total = 0.0

            # Run one full episode using the current policy
            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total += reward
            
            returns.append(total)

        # Return the mean episodic return over all evaluation episodes
        return float(np.mean(returns))

if __name__ == "__main__":
    # Small standalone test: training on "cramped_room"
    student = StudentPPO(verbose=1)
    avg_ret = student.train_on_layout("cramped_room", total_timesteps=50_000, eval_episodes=5)
    print("Average return on cramped_room after 50k steps:", avg_ret)
