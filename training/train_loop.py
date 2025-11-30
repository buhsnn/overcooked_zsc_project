# training/train_loop.py

from typing import List

from teacher.teacher_agent import TeacherAgent
from student.train_ppo_student import StudentPPO


def run_teacher_student_training(
    n_iterations: int = 20,
    train_steps_per_iter: int = 20_000,
):
    """
    Main loop:
    - Teacher chooses a layout.
    - Student trains on it.
    - Teacher updates the scores (regret, novelty, progress).
    """

    teacher = TeacherAgent(
        buffer_size=50,
        w_regret=1.0,
        w_novelty=0.5,
        w_progress=0.5,
        temperature=1.0,
    )
    student = StudentPPO(verbose=1)

    history: List[dict] = []

    for it in range(n_iterations):
        # 1) Teacher chooses a layout
        layout = teacher.sample_layout()

        print(f"\n[ITER {it}] Selected layout by teacher: {layout}")

        # 2) Student trains on this layout
        avg_return = student.train_on_layout(
            layout_name=layout,
            total_timesteps=train_steps_per_iter,
            eval_episodes=5,
        )

        print(f"[ITER {it}] Student avg return on {layout}: {avg_return:.2f}")

        # 3) Teacher updates its stats (regret, progress, etc.)
        teacher.update_after_episode(layout, avg_return)

        # 4) Log some data for analysis
        history.append(
            {
                "iter": it,
                "layout": layout,
                "avg_return": avg_return,
            }
        )

    return history


if __name__ == "__main__":
    history = run_teacher_student_training(
        n_iterations=10,
        train_steps_per_iter=10_000,
    )
    print("\nTraining finished. History:")
    for h in history:
        print(h)
