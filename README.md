# Curator-Guided ACCEL for Zero-Shot Coordination in Overcooked-AI

This project studies **generalization in cooperative multi-agent reinforcement learning (MARL)** using the Overcooked-AI environment.

We propose **Curator-Guided ACCEL**, a framework extending **Unsupervised Environment Design (UED)** by introducing a **Curator mechanism** that prioritizes training environments based on their learning potential.

The goal is to train an agent capable of **zero-shot coordination in unseen task layouts**.

Project developed for **AI611 – Deep Reinforcement Learning** at **KAIST Graduate School of AI**.

---

# Project Overview

In cooperative MARL environments like Overcooked-AI, agents often **overfit to training layouts** and fail when the environment changes.

Small modifications to the kitchen layout can cause **dramatic performance collapse**.

To address this issue, we study **automatic curriculum generation** using **Unsupervised Environment Design (UED)**.

Our contribution introduces a **Curator** that selects training environments based on three learning signals:

- **Regret** – tasks the agent currently fails but could potentially solve  
- **Novelty** – diversity of layouts to avoid mode collapse  
- **Student Progress** – environments where the agent is improving  

This results in a **dynamic curriculum** that focuses training on the most informative environments.

---

# Method

Our framework extends **ACCEL (Evolving Curricula with Regret-Based Environment Design)** by replacing **uniform replay** with **curated replay**.

Each environment θ in the buffer is scored using:

![Results](images/Score.png)

Where:

- **Regret** measures difficulty
- **Novelty** encourages diversity
- **Progress** focuses on environments where learning occurs

The curator samples environments using a **softmax distribution over scores**, ensuring training focuses on the **learning frontier**.

---

# Algorithm Overview

![Results](images/Algorithm.png)


This creates a **co-evolutionary curriculum** between the agent and generated environments.

---

# Environment

We use **Overcooked-AI**, a cooperative grid-world environment where two agents must coordinate to cook and serve soups.

Task sequence:

1. Pick ingredients  
2. Place in pot  
3. Cook soup  
4. Plate  
5. Deliver  

Optimal performance requires:

- coordination
- motion planning
- collision avoidance

---

# Experimental Setup

![Results](Images/Hyperparameter.png)

Student agent:
- PPO (Stable Baselines3)

Partner agent:
- Fixed GreedyHumanModel

Training:
- 1000 iterations
- 1024 steps per iteration

Evaluation metric:
- **Average episode return on unseen layouts**

![Results](Images/Experimental_Setup.png)

---

# Baselines

We compare against three curriculum strategies:

- **Domain Randomization (DR)**
- **Prioritized Level Replay (PLR)**
- **ACCEL**

Our approach:

- **Curator-Guided ACCEL**
![Results](Images/ACCEL_Curator.png)

![Results](Images/Accel.png)


---

# Results

Our method achieves:

- **more stable learning**
- **higher final zero-shot performance**
- **better generalization to unseen layouts**

Compared to Domain Randomization and standard ACCEL.

The Curator mechanism allows training to focus on environments that are:

- difficult
- diverse
- currently learnable

![Results](Images/ZeroShot.png)

---

# Procedural Layout Generation

To scale beyond official Overcooked layouts, we implemented a **procedural layout generator**.

Layouts include:

- randomized interior walls
- random station placement
- strict validation checks



Stations include:

- Onion dispenser
- Pot
- Dish dispenser
- Serving station

Layouts are represented as ASCII grids.


![Results](Images/LayoutGeneration.png)

![Results](Images/Evolution_Mutation.png)


---

# Installation

Clone repository:

- git clone https://github.com/buhsnn/overcooked_zsc_project.git

- cd overcooked_zsc_project



---

# Running Experiments

Domain Randomization baseline:
- bash main_dr.sh


Curator-Guided ACCEL:
- bash main_ours.sh

---

# Visualization

You can visualize layouts and logs with:
- python vis_grid.py
- python vis_logs.py


---

# Paper

This repository accompanies the research report:

**Sequential Teacher for Unsupervised Environment Design in Overcooked-AI**

AI611 – Deep Reinforcement Learning  
KAIST Graduate School of AI

Authors:

- Bushra Monika Hossain  
- Daehyun We  
- Yujin Bae  

---

# Future Work

Possible extensions include:

- scaling the buffer to >10,000 environments
- co-evolving partner policies
- applying the framework to other MARL environments

---

# Acknowledgments

This project builds upon:

- Overcooked-AI
- Stable Baselines3
- ACCEL and UED research frameworks

- 
