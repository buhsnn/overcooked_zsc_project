import numpy as np
from stable_baselines3 import PPO

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action


class PPOStudentAgent(Agent):
    """
    Wrapper SB3 PPO -> Agent Overcooked compatible.

    - model : PPO (stable-baselines3)
    - featurize_fn : (state) -> (obs_p0, obs_p1)
    - agent_index : 0 ou 1
    """

    def __init__(self, model: PPO, featurize_fn, agent_index: int):
        super().__init__()
        self.model = model
        self.featurize_fn = featurize_fn
        self.agent_index = agent_index
        self.mdp = None

    def reset(self):
        
        pass

    def set_mdp(self, mdp):
        self.mdp = mdp

    def action(self, state):
        
        obs_p0, obs_p1 = self.featurize_fn(state)

        obs = obs_p0 if self.agent_index == 0 else obs_p1

        
        obs_batch = obs.reshape(1, -1)

        action_idx, _ = self.model.predict(obs_batch, deterministic=True)

        if hasattr(action_idx, "item"):
            action_idx = int(action_idx)

        oc_action = Action.INDEX_TO_ACTION[action_idx]

        return oc_action, {}
