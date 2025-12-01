"""
Launch a full teacher-student training experiment.
"""

import debugpy

from student.train_ppo_student import StudentPPO
from teacher.teacher_agent import TeacherAgent
from training.trainer import Trainer

if __name__ == "__main__":
    # debugpy.listen(5678)
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()
    # print("Debugger attached.")
    
    trainer = Trainer(n_iterations=50, buffer_size=3)
    trainer.run()
    trainer.eval()
