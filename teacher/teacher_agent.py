"""
Teacher agent using REINFORCE to generate layouts maximizing:
J = α LP + β Novelty - γ Unsolvability
"""

class TeacherAgent:
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.layout_history = []

    def sample_layout(self):
        """
        Propose a new layout candidate (string or grid).
        """
        raise NotImplementedError("Implement layout generation here.")

    def update(self, learning_progress, novelty, unsat):
        """
        Update policy parameters (REINFORCE gradient step).
        """
        raise NotImplementedError("Implement REINFORCE update here.")
