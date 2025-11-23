"""
Main training loop sequencing:
teacher → student → teacher updates
"""

class Trainer:
    def __init__(self, teacher, student):
        self.teacher = teacher
        self.student = student

    def run(self, iterations=50):
        for i in range(iterations):
            layout = self.teacher.sample_layout()
            print(f"[ITER {i}] Teacher proposed layout:", layout)

            # Train PPO on generated layout
            avg_return = self.student.train_on_layout(layout)

            # Compute metrics
            novelty = 0.0
            unsat = 0.0
            lp = avg_return  # placeholder

            # Update teacher
            self.teacher.update(lp, novelty, unsat)

            print(f" → LP={lp:.2f}, Novelty={novelty:.2f}, Unsat={unsat:.2f}")

