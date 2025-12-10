"""
Launch a full teacher-student training experiment.
"""

import argparse

import debugpy

from student.train_ppo_student import StudentPPO
from teacher.teacher_agent import TeacherAgent
from training.trainer import Trainer

if __name__ == "__main__":
    # debugpy.listen(5678)
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()
    # print("Debugger attached.")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iterations", type=int, default=1000, help="Number of training iterations.")
    parser.add_argument("--train_steps_per_iter", type=int, default=1_000, help="Training steps per iteration.")
    parser.add_argument("--buffer_size", type=int, default=10, help="Teacher buffer size.")
    parser.add_argument("--w_regret", type=float, default=0.01, help="Weight for regret.")
    parser.add_argument("--w_novelty", type=float, default=0.03, help="Weight for novelty.")
    parser.add_argument("--w_progress", type=float, default=-0.1, help="Weight for student progress.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")
    parser.add_argument("--s_threshold", type=float, default=2.0, help="Score threshold for adding layouts to buffer.")
    parser.add_argument("--student_verbose", type=int, default=1, help="Verbosity level for student training.")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save logs.")
    args = parser.parse_args()
    
    trainer = Trainer(
        n_iterations=args.n_iterations,
        train_steps_per_iter=args.train_steps_per_iter,
        buffer_size=args.buffer_size,
        w_regret=args.w_regret,
        w_novelty=args.w_novelty,
        w_progress=args.w_progress,
        temperature=args.temperature,
        s_threshold=args.s_threshold,
        student_verbose=args.student_verbose,
        log_dir=args.log_dir,
    )
    trainer.run()
    trainer.eval()
