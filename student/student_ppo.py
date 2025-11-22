"""
PPO student agent.
Trains on a single Overcooked layout.
"""

class StudentPPO:
    def __init__(self, config=None):
        self.config = config

    def train_on_layout(self, layout_name, timesteps=20000):
        """
        Returns average return achieved by the PPO agent on this layout.
        """
        raise NotImplementedError("Implement PPO training here.")
