import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

import manipulator_mujoco
from model.actor_critic import ActorCritic
from memory.rollout_buffer import RolloutBuffer

def make_env(env_id="manipulator_mujoco/AuboI5Env-v0", render_mode=None):
    return gym.make(env_id, render_mode=render_mode)

def ppo_train(
    env_id="manipulator_mujoco/AuboI5Env-v0",
    total_steps=50_000,
    rollout_len=2048,
    update_epochs=10,
    minibatch_size=256,
    gamma=0.99,
    lam=0.95,
    clip_ratio=0.2,
    vf_coef=0.5,
    ent_coef=0.0,
    lr=3e-4,
    device="cpu",
    render=False,
    save_dir=None,
):
    env = make_env(env_id, render_mode="rgb_array" if render else None)
    obs, _ = env.reset()
    obs_dim = int(np.prod(obs.shape))
    act_dim = int(np.prod(env.action_space.shape))

    ac = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(ac.parameters(), lr=lr)

    def to_t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    step_count = 0
    rewards_log = []
    best_ret = -1e9
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    while step_count < total_steps:
        curr_len = int(min(rollout_len, total_steps - step_count))
        buf = RolloutBuffer(curr_len, obs_dim, act_dim, gamma, lam)
        obs, _ = env.reset()
        ep_len = 0
        for t in range(curr_len):
            o = to_t(obs).unsqueeze(0)
            with torch.no_grad():
                action, logp, dist = ac.act(o)
                value = ac.value(o)
            act_np = action.squeeze(0).cpu().numpy()
            act_np = np.clip(act_np, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, info = env.step(act_np)
            buf.store(obs.astype(np.float32), act_np.astype(np.float32), float(logp.item()), float(reward), float(terminated or truncated), float(value.item()))
            obs = next_obs
            ep_len += 1
            step_count += 1
            if terminated or truncated:
                last_val = 0.0
                if not terminated:
                    with torch.no_grad():
                        last_val = float(ac.value(to_t(obs).unsqueeze(0)).item())
                buf.finish_path(last_val)
                obs, _ = env.reset()
                ep_len = 0
            if step_count >= total_steps:
                break

        if buf.ptr > buf.path_start_idx:
            with torch.no_grad():
                last_val = float(ac.value(to_t(obs).unsqueeze(0)).item())
            buf.finish_path(last_val)

        data = buf.get()
        N = data["obs"].shape[0]
        obs_t = to_t(data["obs"])  # [N, obs_dim]
        acts_t = to_t(data["acts"])  # [N, act_dim]
        advs_t = to_t(data["advs"])  # [N]
        rets_t = to_t(data["rets"])  # [N]
        old_logps_t = to_t(data["logps"])  # [N]

        idxs = np.arange(N)
        for epoch in range(update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, N, minibatch_size):
                mb = idxs[start:start+minibatch_size]
                mb_obs = obs_t[mb]
                mb_acts = acts_t[mb]
                mb_advs = advs_t[mb]
                mb_rets = rets_t[mb]
                mb_old_logps = old_logps_t[mb]

                mean, std = ac.actor(mb_obs)
                dist = torch.distributions.Normal(mean, std)
                logps = dist.log_prob(mb_acts).sum(-1)
                ratio = torch.exp(logps - mb_old_logps)
                clip_obj = torch.min(ratio * mb_advs, torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * mb_advs)
                policy_loss = -clip_obj.mean()

                values = ac.value(mb_obs)
                value_loss = F.mse_loss(values, mb_rets)

                entropy = dist.entropy().sum(-1).mean()

                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                optimizer.step()

        ep_reward = float(np.sum(data["rets"]))
        rewards_log.append(ep_reward)
        if save_dir and (ep_reward > best_ret):
            best_ret = ep_reward
            path = os.path.join(save_dir, "best.pt")
            torch.save({
                "model": ac.state_dict(),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "cfg": {
                    "gamma": gamma, "lam": lam, "clip_ratio": clip_ratio,
                    "vf_coef": vf_coef, "ent_coef": ent_coef, "lr": lr
                }
            }, path)

    if save_dir:
        torch.save({"model": ac.state_dict()}, os.path.join(save_dir, "final.pt"))
        np.save(os.path.join(save_dir, "rewards.npy"), np.array(rewards_log, dtype=np.float32))
    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="manipulator_mujoco/AuboI5Env-v0")
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    ppo_train(env_id=args.env_id, total_steps=args.total_steps, render=args.render)

if __name__ == "__main__":
    main()
