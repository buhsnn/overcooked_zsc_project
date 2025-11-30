# Overcooked Zero-Shot Coordination Project

This project implements a **Student–Teacher learning pipeline** on top of the **Overcooked-AI** environment (HumanCompatibleAI).  
Because Overcooked-AI is not available on pip, the environment must be installed manually before training the RL agent.

The objective:  
- Build a **Gym wrapper** around Overcooked-AI  
- Train a **PPO student agent** using Stable-Baselines3  
- Later integrate a **teacher policy** to guide learning and evaluate zero-shot coordination  

---

# 1. Installation

## 1.1 Install Overcooked-AI (source only)
Overcooked-AI is not on pip → you must clone the repository manually:

```bash
git clone https://github.com/HumanCompatibleAI/overcooked_ai.git
cd overcooked_ai
pip install -e .
```
## 1.2 Install required dependencies
pip install numpy gymnasium pygame opencv-python tensorflow

## 1.3 Install Stable-Baselines3 (RL algorithms)
pip install stable-baselines3[extra]


# 2. Project Structure

overcooked_zsc_project/
│
│── main.py
│
├── env/
│   ├── overcooked_wrapper.py       # Gym wrapper around Overcooked-AI
│   └── __init__.py
│
├── student/
│   ├── train_ppo_student.py        # Training script for the PPO Student
│   └── __init__.py
│
├── teacher/
│   ├── teacher_agent.py            # Placeholder teacher policy
│   └── __init__.py
│
├── training/
│   ├── train_loop.py               # Possible custom training loop (Student–Teacher)
│   └── __init__.py
│
└── utils/
    ├── layout_utils.py             # Layout / environment helpers
    └── __init__.py


# 3. Gym Wrapper (env/overcooked_wrapper.py)

Overcooked-AI is not a Gym environment by default
We implement a wrapper OvercookedGym that turns it into a standard:
observation_space → numerical feature vector (≈ 96 features)

action_space = Discrete(6)
Maps to Overcooked actions:

0 → North

1 → South

2 → East

3 → West

4 → Stay

5 → Interact

For simplicity, both agents receive the same action (single-agent control).

This makes PPO training possible.

# 4. Training the PPO Student
The student is trained using Stable-Baselines3 PPO.

RUN the NOTEBOOK OR do this command:
```bash
python student/train_ppo_student.py
```

The script:

- Creates the environment via OvercookedGym("cramped_room")

- Builds a PPO agent (MlpPolicy)

- Trains for 200k timesteps

- Saves the model to:ppo_student.zip

 # 5.Testing the Trained Student
```bash
 from stable_baselines3 import PPO
from env.overcooked_wrapper import OvercookedGym

env = OvercookedGym("cramped_room")
model = PPO.load("ppo_student", env=env)

obs, _ = env.reset()
for _ in range(10):
    action, _ = model.predict(obs)
    obs, reward, done, trunc, info = env.step(action)
    print(action, reward)
```

# 6. Overcooked Layouts

Layout files are located in:
overcooked_ai_py/data/layouts/*.layout
Example layout (cramped_room.layout):

XXPXX
O   O
X   X
XDXSX

Legend:

-X = Wall
-P = Pot
-O = Onion source
-D = Dish dispenser
-S = Serving station

You can display all available layouts in the notebook by listing the .layout files.


# 7. Notes
- PPO initially performs poorly (reward ≈ 0): Overcooked-AI is a hard coordination problem.
- The MediumLevelActionManager is recomputed at reset → slow first launch.
- A teacher policy (planned next step) can accelerate learning in zero-shot settings.



- Evaluate zero-shot coordination with unseen partners
