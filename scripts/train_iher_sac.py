# train_iher_sac.py
import os
import gymnasium as gym
import gymnasium_robotics
import argparse
import manipulator_mujoco
import datetime

from algorithm.IHER_SAC import IHER_SAC

def make_env(env_id="FetchPickAndPlace-v4"):
    return gym.make(env_id)

def train(env_id: str = "FetchPickAndPlace-v4",
          total_timesteps: int = 1_000_000,
          buffer_size:int = int(1e6),
          learning_rate: float = 1e-3,
          batch_size: int = 512,
          tau: float = 0.05,
          gamma: float = 0.95,
          seed: int = 42,
          device: str = "cuda",
          ):
    env = make_env(env_id)
    # create model: default policy MultiInputPolicy is suitable for GoalEnv dict obs
    policy_kwargs = dict(
        net_arch=[256, 256],
    )
    
    model = IHER_SAC(
        "MultiInputPolicy",
        env=env,
        verbose=1,
        buffer_size=buffer_size,
        learning_rate=learning_rate,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        ent_coef="auto",
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=device,
        tensorboard_log="./logs/iher_sac",
    )
    model.learn(total_timesteps=total_timesteps, log_interval=10, progress_bar=True)

    # save
    time_now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model.save(f"checkpoints/iher_sac/{env_id}/{time_now}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="FetchPickAndPlace-v4")
    parser.add_argument("--env", type=str)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train(
        env_id=args.env if args.env else args.env_id,
        buffer_size=args.buffer_size,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        seed=args.seed,
        device=args.device,
    )
