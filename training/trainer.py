# training/trainer.py

import time
from typing import List, Dict

from teacher.teacher_agent import TeacherAgent
from student.train_ppo_student import StudentPPO


class Trainer:
    """
    Clean trainer for Teacher–Student (UED).
    """

    def __init__(
        self,
        n_iterations: int = 20,
        train_steps_per_iter: int = 20_000,
        buffer_size: int = 50,
        w_regret: float = 1.0,
        w_novelty: float = 0.5,
        w_progress: float = 0.5,
        temperature: float = 1.0,
        student_verbose: int = 1,
    ):
        self.n_iterations = n_iterations
        self.train_steps_per_iter = train_steps_per_iter

        # Teacher
        self.teacher = TeacherAgent(
            buffer_size=buffer_size,
            w_regret=w_regret,
            w_novelty=w_novelty,
            w_progress=w_progress,
            temperature=temperature,
        )

        # Student (new version)
        self.student = StudentPPO(verbose=student_verbose)

        # Log history
        self.history: List[Dict] = []

    # ---------------------------------------------------------------------- #
    def run(self):
        """
        Run the loop teacher → student → feedback.
        """
        print("\n===== START TRAINING =====\n")

        for it in range(self.n_iterations):

            print("=" * 50)
            print(f"[ITERATION {it}]")

            # ---------------------------
            # 1) Teacher chooses a layout
            # ---------------------------
            layout = self.teacher.sample_layout()
            print(f"[Teacher] Selected layout: {layout}")

            # ---------------------------
            # 2) Student trains its PPO on this layout
            # ---------------------------
            avg_return = self.student.train_on_layout(
                layout_name=layout,
                total_timesteps=self.train_steps_per_iter,
                eval_episodes=5,
            )
            print(f"[Student] Avg return on {layout}: {avg_return:.2f}")

            # ---------------------------
            # 3) Teacher updates its stats
            # ---------------------------
            self.teacher.update_after_episode(layout, avg_return)

            # ---------------------------
            # 4) Log
            # ---------------------------
            self.history.append(
                {
                    "iteration": it,
                    "layout": layout,
                    "avg_return": avg_return,
                    "timestamp": time.time(),
                }
            )

        print("\n===== TRAINING FINISHED =====\n")
        return self.history
