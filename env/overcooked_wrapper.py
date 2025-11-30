import gymnasium as gym
import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action

from teacher.teacher_pair import make_teacher_env_and_pair


class OvercookedGym(gym.Env):
    """
    SB3-compatible environment where:

    - PPO controls ONE agent (agent_index = 0 or 1)
    - Partner is a fixed TEACHER (GreedyHumanModel)
    """

    metadata = {"render_modes": []}

    def __init__(self, layout_name="cramped_room", horizon=400, agent_index=0):
        super().__init__()
        assert agent_index in [0, 1], "agent_index must be 0 or 1"
        self.agent_index = agent_index

        # ========================
        # Base Overcooked Env
        # ========================
        self.mdp = OvercookedGridworld.from_layout_name(
            layout_name,
            old_dynamics=True,
        )
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)
        self.env.reset()

        # ========================
        # Teacher partner (GreedyHumanModel)
        # ========================
        _, teacher_pair = make_teacher_env_and_pair(layout_name, horizon)
        self.teacher0 = teacher_pair.agents[0]
        self.teacher1 = teacher_pair.agents[1]

        # ========================
        # Observation space
        # ========================
        obs_p0, obs_p1 = self.env.featurize_state_mdp(self.env.state)
        my_obs = obs_p0 if self.agent_index == 0 else obs_p1

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=my_obs.shape,
            dtype=np.float32,
        )

        # ========================
        # Action space (6 actions)
        # ========================
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))

    # ----------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()

        obs_p0, obs_p1 = self.env.featurize_state_mdp(self.env.state)
        my_obs = obs_p0 if self.agent_index == 0 else obs_p1

        return my_obs.astype(np.float32), {}

    # ----------------------------------------------------

    def step(self, action):

        # Convert numpy scalar to int
        if hasattr(action, "item"):
            action = int(action)

        # Student action
        my_action = Action.INDEX_TO_ACTION[action]

        # Teacher actions
        teacher_a0, _ = self.teacher0.action(self.env.state)
        teacher_a1, _ = self.teacher1.action(self.env.state)

        # Joint action: student on one side, teacher on the other
        if self.agent_index == 0:
            joint_action = (my_action, teacher_a1)
        else:
            joint_action = (teacher_a0, my_action)

        # Step env
        next_state, reward, done, info = self.env.step(joint_action)

        # Featurize
        obs_p0, obs_p1 = self.env.featurize_state_mdp(next_state)
        my_obs = obs_p0 if self.agent_index == 0 else obs_p1

        # Gymnasium API: (obs, reward, terminated, truncated, info)
        return my_obs.astype(np.float32), reward, done, False, info
