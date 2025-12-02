import gymnasium as gym
import gymnasium_robotics
import argparse
import os
from tqdm import tqdm
import imageio
import numpy as np
import random
import sys
import types
from stable_baselines3 import SAC, PPO
from algorithm.IHER_SAC import IHER_SAC

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

def eval_video(env_id: str = "FetchPickAndPlace-v4",
               checkpoint_name: str = "202511281553",
               algo: str = "iher_sac",
               num_episodes: int = 100,
               render_mode: str = "rgb_array",
               base_seed: int | None = None,
               ):    
    env = gym.make(env_id, render_mode=render_mode)
    model_path = os.path.join("checkpoints", algo, env_id, checkpoint_name)
    if algo == "iher_sac":
        model = IHER_SAC.load(model_path, env=env)
    elif algo == "her_sac":
        model = SAC.load(model_path, env=env)
    elif algo == "sac":
        model = SAC.load(model_path, env=env)
    elif algo == "ppo":
        model = PPO.load(model_path, env=env)
    else:
        raise ValueError(f"Unknown algo: {algo}")
    
    success_count = 0
    goals_record = []
    sample_frames = []
    count = 0
    for episode in tqdm(range(num_episodes)):
        obs, info = env.reset(seed=random.randint(0, 2**31 - 1))
        done = False
        truncated = False
        frames = []
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            frame = env.render()
            frames.append(frame)
            is_success = info["is_success"]
            if is_success:
                success_count += 1
        count += 1
        if count % 10 == 0:
            sample_frames.append(frames)
            
    success_rate = success_count / num_episodes

    output_dir = os.path.join("output", algo, env_id) + f"_success_rate_{success_rate:.2f}"
    os.makedirs(output_dir, exist_ok=True)
    for i, frames in enumerate(sample_frames):
        imageio.mimsave(os.path.join(output_dir, f"sample_{i}.gif"), frames, fps=30, loop=0)
    print(f"Saved all output videos to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="FetchPickAndPlace-v4")
    parser.add_argument("--checkpoint_name", type=str, default="202511281553")
    parser.add_argument("--algo", type=str, default="iher_sac")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--base_seed", type=int, default=None)
    args = parser.parse_args()
    
    eval_video(**vars(args))
    
        
