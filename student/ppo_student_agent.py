import os
from stable_baselines3 import PPO
from env.overcooked_wrapper import OvercookedGym


class PPOStudentAgent:
    """
    Custom wrapper for a student PPO agent.
    Exposes the following methods:
        - train_on_layout(layout, timesteps, eval_episodes)
        - evaluate(layout, episodes)
    """

    def __init__(
        self,
        model_path: str = None,
        verbose: int = 1
    ):
        self.verbose = verbose
        self.model_path = model_path

        # PPO model (initialized as None)
        self.model = None

    # -------------------------------------------------------------- #
    def _init_model(self, layout_name: str):
        """Initialize PPO for a given layout."""
        env = OvercookedGym(layout_name, agent_index=0)

        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=self.verbose,
        )

    # -------------------------------------------------------------- #
    def train_on_layout(self, layout_name: str, total_timesteps: int, eval_episodes: int = 5):
        """
        1. Initialize PPO on this layout
        2. Train PPO for a specified number of timesteps
        3. Evaluate PPO performance
        """
        if self.verbose:
            print(f"\n[Student] Training PPO on layout: {layout_name}")

        # Initialize the PPO model
        self._init_model(layout_name)

        # Train the agent
        self.model.learn(total_timesteps=total_timesteps)

        # Evaluate the trained model
        avg_return = self.evaluate(layout_name, eval_episodes)
        return avg_return

    # -------------------------------------------------------------- #
    def evaluate(self, layout_name: str, episodes: int = 5):
        """Evaluate PPO policy on a given number of episodes."""
        env = OvercookedGym(layout_name, agent_index=0)

        total_reward = 0
        for _ in range(episodes):
            obs, _ = env.reset()
            done, truncated = False, False

            while not (done or truncated):
                # Get action prediction from the trained model
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                total_reward += reward

        avg_return = total_reward / episodes
        if self.verbose:
            print(f"[Student] Evaluation return on {layout_name}: {avg_return:.2f}")

        return avg_return
