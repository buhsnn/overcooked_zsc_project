import gymnasium as gym
import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action


class OvercookedGym(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, layout_name="cramped_room", horizon=400):
        super().__init__()

        # Create MDP + Env
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)

        # Initial reset
        self.env.reset()

        # Compute initial observation
        obs_p0, obs_p1 = self.mdp.featurize_state(self.env.state, self.env.mlam)

        # Gym API spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_p0.shape,
            dtype=np.float32
        )

        # 6 discrete actions (converted later to Overcooked actions)
        self.action_space = gym.spaces.Discrete(6)

    def reset(self, seed=None, options=None):
        self.env.reset()
        obs_p0, obs_p1 = self.mdp.featurize_state(self.env.state, self.env.mlam)
        return obs_p0, {}

    def step(self, action):
        # Convert Gym discrete action → Overcooked action
        oc_action = Action.INDEX_TO_ACTION[action]

        # Apply same action for both agents for simplicity
        next_state, reward, done, info = self.env.step((oc_action, oc_action))

        # Featurize next state
        obs_p0, obs_p1 = self.mdp.featurize_state(next_state, self.env.mlam)

        return obs_p0, reward, done, False, info
