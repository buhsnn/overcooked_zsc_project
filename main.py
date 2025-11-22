"""
Launch a full teacher-student training experiment.
"""

from teacher.teacher_agent import TeacherAgent
from student.train_ppo_student import StudentPPO
from training.train_loop import Trainer

if __name__ == "__main__":
    teacher = TeacherAgent()
    student = StudentPPO()

    trainer = Trainer(teacher, student)
    trainer.run(iterations=5)
