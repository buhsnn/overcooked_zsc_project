# training/trainer.py

import datetime
import json
import random
import time
from pathlib import Path
from typing import Dict, List

from student.train_ppo_student import StudentPPO
from teacher.teacher_agent import TeacherAgent
from utils.layout_utils import EVAL_LAYOUTS


class Trainer:
    """
    Clean trainer for Teacher–Student (UED).
    """

    def __init__(
        self,
        n_iterations: int = 20,
        train_steps_per_iter: int = 1_000,
        buffer_size: int = 50,
        w_regret: float = 0.01,
        w_novelty: float = 0.5,
        w_progress: float = -0.1,
        temperature: float = 1.0,
        s_threshold: float = 2.0,
        student_verbose: int = 1,
        log_dir: str = "./logs",
    ):
        self.n_iterations = n_iterations
        self.train_steps_per_iter = train_steps_per_iter
        self.s_threshold = s_threshold

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
        self.log_dir = Path(log_dir) / f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_tspi{train_steps_per_iter}_bs{buffer_size}_wr{w_regret}_wn{w_novelty}_wp{w_progress}_temp{temperature}_sth{s_threshold}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------- #
    def run(self):
        """
        Run the loop teacher → student → feedback.
        """
        print("\n===== START TRAINING =====\n")

        for it in range(self.n_iterations):

            print("=" * 50)
            print(f"[ITERATION {it}]")
            
            score_snapshot = self.teacher.get_score_snapshot()
            with open(self.log_dir / f"score_snapshot_iter{it}.json", "w") as f:
                json.dump(score_snapshot, f, indent=4)

            if random.random() < 0.5:
                # ---------------------------
                # 1) Generate random layout
                # ---------------------------
                layout = self.teacher.generate_layout()
                print(f"[Teacher] Generated layout: {layout}")
                
                # ---------------------------
                # 2) Evaluate agent without training
                # ---------------------------
                avg_return = self.student.train_on_layout(
                    layout_name=layout,
                    total_timesteps=0,
                    eval_episodes=5,
                )
                
                # ---------------------------
                # 3) Compute score of layout, and add to buffer
                # ---------------------------
                score = self.teacher.compute_score(layout, [avg_return])
                if score >= self.s_threshold:
                    print(f"[Teacher] Adding layout {layout} to buffer with score {score:.2f}")
                    self.teacher.update_after_episode_wo_mutate(layout, avg_return)
                
            else:
                # ---------------------------
                # 1) Teacher chooses a layout from buffer
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
                self.teacher.update_after_episode_wo_mutate(layout, avg_return)
                
                # TODO: After mid report, add mutation
                # self.teacher.update_after_episode(layout, avg_return)

            # ---------------------------
            # 4) Validation
            # ---------------------------
            self.eval()

            # ---------------------------
            # 5) Log
            # ---------------------------
            self.history.append(
                {
                    "iteration": it,
                    "layout": layout,
                    "avg_return": avg_return,
                    "timestamp": time.time(),
                }
            )
            with open(self.log_dir / "train.json", "w") as f:
                json.dump(self.history, f, indent=4)
            

        print("\n===== TRAINING FINISHED =====\n")
        return self.history


    def eval(self):
        avg_return_dict = {}
        for layout in EVAL_LAYOUTS:
            avg_return = self.student.train_on_layout(
                layout_name=layout,
                total_timesteps=0,
                eval_episodes=5,
            )
            avg_return_dict[layout] = avg_return
        avg_return_dict["overall_avg"] = sum(avg_return_dict.values()) / len(avg_return_dict)
        
        # Save as json
        with open(self.log_dir / f"eval_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json", "w") as f:
            json.dump(avg_return_dict, f, indent=4)
        
        return avg_return_dict
        