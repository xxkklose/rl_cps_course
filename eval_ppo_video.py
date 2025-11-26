import argparse
import os
import numpy as np
import gymnasium as gym
import torch
import imageio.v2 as imageio

import manipulator_mujoco
from model.actor_critic import ActorCritic



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--env_id', type=str, default='manipulator_mujoco/AuboI5Env-v0')
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--out', type=str, default='eval.mp4')
    args = parser.parse_args()

    ckpt = torch.load(args.weights, map_location='cpu')
    obs_dim = ckpt.get('obs_dim')
    act_dim = ckpt.get('act_dim')

    ac = ActorCritic(obs_dim, act_dim)
    ac.load_state_dict(ckpt['model'])
    ac.eval()

    env = gym.make(args.env_id, render_mode='rgb_array')
    obs,_ = env.reset()
    frames = []
    for t in range(args.steps):
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mean, std = ac.actor(o)
            action = mean.squeeze(0).cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            obs,_ = env.reset()

    env.close()

    if len(frames) > 0:
        try:
            imageio.mimsave(args.out, frames, fps=int(1.0/env.unwrapped._timestep))
            print('video_saved', args.out, len(frames))
        except Exception as e:
            np.save(os.path.splitext(args.out)[0] + '.npy', np.array(frames))
            print('video_saved_npy_fallback', e)

if __name__ == '__main__':
    main()
