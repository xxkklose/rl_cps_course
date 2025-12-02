import os
import argparse
import imageio
import torch
import gymnasium as gym
import gymnasium_robotics
import manipulator_mujoco
import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import ProgressBarCallback

os.environ.setdefault("MUJOCO_GL", "egl")

def make_train_env(env_id="FetchPickAndPlace-v4", n_envs=4, seed=42):
    env = make_vec_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
    )
    env = VecMonitor(env)
    return env

def make_test_env(env_id="FetchPickAndPlace-v4"):
    return gym.make(env_id, render_mode="rgb_array")


def train_ppo(
    env_id: str = "FetchPickAndPlace-v4",
    seed: int = 42,
    total_timesteps: int = 1_000_000,
    learning_rate: float = 3e-4,
    batch_size: int = 2048,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.0,
    device: str = "cuda",
):
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    env = make_train_env(n_envs=4, seed=seed)

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[dict(pi=[256, 128], vf=[256, 128])]
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        n_steps=2048,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        device=device,
        verbose=1,
        tensorboard_log="./logs/ppo",
    )

    model.learn(
        total_timesteps=1_000_000,
        progress_bar=True,                
        callback=ProgressBarCallback(),   
    )

    time_now = datetime.now().strftime("%Y%m%d%H%M")
    model.save(f"checkpoints/ppo/{env_id}/{time_now}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", "-e", type=str, default="FetchPickAndPlace-v4")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--total_timesteps", "-t", type=int, default=1_000_000)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", "-b", type=int, default=2048)
    parser.add_argument("--gamma", "-g", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--device", "-d", type=str, default="cuda")

    args = parser.parse_args()

    train_ppo(
        env_id=args.env_id,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        device=args.device,
    )
