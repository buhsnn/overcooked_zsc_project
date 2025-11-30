from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from env.overcooked_wrapper import OvercookedGym


def make_env():
   
    return OvercookedGym(layout_name="cramped_room", horizon=400)


if __name__ == "__main__":
    
    vec_env = DummyVecEnv([make_env])
    vec_env = VecMonitor(vec_env)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        verbose=1,
    )

    
    model.learn(total_timesteps=200_000)

    model.save("ppo_student")
    print("Training finished (self-play simple).")
