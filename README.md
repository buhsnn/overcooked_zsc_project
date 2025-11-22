1. Installation (Before Anything Else)

Overcooked-AI is not available on pip, it must be installed manually.

1.1 : Clone & install Overcooked-AI
git clone https://github.com/HumanCompatibleAI/overcooked_ai.git
cd overcooked_ai
pip install -e .


1.2. Install Overcooked-AI dependencies

pip install numpy gymnasium pygame opencv-python tensorflow

1.3 Install RL algorithms (Stable-Baselines3):  This gives us PPO, A2C, training utilities, etc.
pip install stable-baselines3[extra]

