import gymnasium as gym
import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action


class OvercookedGym(gym.Env):
    """
    SB3-compatible wrapper around the official OvercookedEnv.

    - use OvercookedGridworld.from_layout_name(..., old_dynamics=True)
    - use env.featurize_state_mdp (same features as BC / PPO rllib)
    - Observation = features of the agent 0 only (vector 1D)
    - Action = Discrete(6) → map into Action.INDEX_TO_ACTION
    """

    metadata = {"render_modes": []}

    def __init__(self, layout_name="cramped_room", horizon=400):
        super().__init__()

        # MDP official
        self.mdp = OvercookedGridworld.from_layout_name(
            layout_name,
            old_dynamics=True
        )

        # Env official OvercookedEnv
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)

        # Reset initial 
        self.env.reset()

        # *** OBSERVATIONS ***
        # Use oficial function of featurization
        # featurize_state_mdp(state) -> (obs_p0, obs_p1)
        obs_p0, obs_p1 = self.env.featurize_state_mdp(self.env.state)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_p0.shape,
            dtype=np.float32,
        )

        # *** ACTIONS ***
        # 6 actions (N, S, E, W, STAY, INTERACT) →  0..5
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))

    def reset(self, seed=None, options=None):
        # Reset env officiel
        self.env.reset()
        obs_p0, obs_p1 = self.env.featurize_state_mdp(self.env.state)
        return obs_p0, {}

    def step(self, action):
        """
        action : int ∈ [0..5] (SB3)
        We convertit it to  Action.INDEX_TO_ACTION[action],
        then we apply the same action to the two agents (student self-play simple).
        """

        # Sécurity
        if hasattr(action, "item"):
            action = int(action)

        oc_action = Action.INDEX_TO_ACTION[action]

        
        next_state, reward, done, info = self.env.step(
            (oc_action, oc_action)
        )

        
        obs_p0, obs_p1 = self.env.featurize_state_mdp(next_state)

        # Gymnasium API 
        return obs_p0, reward, done, False, info
