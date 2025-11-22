from stable_baselines3 import PPO
from env.overcooked_wrapper import OvercookedGym

# Create env
env = OvercookedGym("cramped_room")

# PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
)

# Training
model.learn(total_timesteps=200_000)

model.save("ppo_student")
print("Training finished!")
