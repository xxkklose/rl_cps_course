import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import datetime
from tqdm import tqdm

import manipulator_mujoco
from model.actor_critic import ActorCritic
from memory.rollout_buffer import RolloutBuffer

def make_env(env_id="manipulator_mujoco/AuboI5Env-v0", render_mode=None):
    return gym.make(env_id, render_mode=render_mode)

def ppo_train(
    env_id="manipulator_mujoco/AuboI5Env-v0",
    total_steps=50000,
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

    try:
        act_low = env.action_space.low.astype(np.float32)
        act_high = env.action_space.high.astype(np.float32)
    except Exception:
        act_low = None
        act_high = None
    ac = ActorCritic(obs_dim, act_dim, act_low, act_high).to(device)
    optimizer = optim.Adam(ac.parameters(), lr=lr)

    def to_t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    step_count = 0
    rewards_log = []
    best_ret = -1e9
    save_dir += f"/{datetime.datetime.now().strftime('%Y%m%d%H%M')}"
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    pbar = tqdm(total=total_steps, desc="Training PPO", ncols=200)

    # running observation normalization (updated each rollout)
    obs_mean = torch.zeros(obs_dim, dtype=torch.float32, device=device)
    obs_std = torch.ones(obs_dim, dtype=torch.float32, device=device)
    def norm_obs(o_t):
        out = (o_t - obs_mean) / (obs_std + 1e-8)
        out = torch.nan_to_num(out)
        return torch.clamp(out, -10.0, 10.0)
    while step_count < total_steps:
        curr_len = int(min(rollout_len, total_steps - step_count))
        buf = RolloutBuffer(curr_len, obs_dim, act_dim, gamma, lam)
        obs, _ = env.reset()
        ep_len = 0
        ep_reward_sum = 0.0
        last_loss = 0.0
        ep_successes = 0

        for t in range(curr_len):
            o = to_t(obs).unsqueeze(0)
            with torch.no_grad():
                action, logp, dist = ac.act(norm_obs(o))
                value = ac.value(norm_obs(o))
            act_np = action.squeeze(0).cpu().numpy()
            act_np = np.clip(act_np, env.action_space.low, env.action_space.high)
            with torch.no_grad():
                logp = dist.log_prob(to_t(act_np).unsqueeze(0)).sum(-1)
            next_obs, reward, terminated, truncated, info = env.step(act_np)
            buf.store(obs.astype(np.float32), act_np.astype(np.float32), float(logp.item()), float(reward), float(terminated or truncated), float(value.item()))
            obs = next_obs
            ep_reward_sum += reward
            ep_len += 1
            step_count += 1
            pbar.update(1)
            if isinstance(info, dict) and info.get("success"):
                ep_successes += 1

            if terminated or truncated:
                last_val = 0.0
                if not terminated:
                    with torch.no_grad():
                        last_val = float(ac.value(norm_obs(to_t(obs).unsqueeze(0))).item())
                buf.finish_path(last_val)
                obs, _ = env.reset()
                ep_len = 0
            if step_count >= total_steps:
                break

        if buf.ptr > buf.path_start_idx:
            with torch.no_grad():
                last_val = float(ac.value(norm_obs(to_t(obs).unsqueeze(0))).item())
            buf.finish_path(last_val)

        data = buf.get()
        N = data["obs"].shape[0]
        # update running obs stats for next rollout
        obs_mean = to_t(data["obs"].mean(axis=0))
        obs_std = to_t(data["obs"].std(axis=0) + 1e-8)

        obs_t = to_t(data["obs"])  # [N, obs_dim]
        acts_t = to_t(data["acts"])  # [N, act_dim]
        advs_t = to_t(data["advs"])  # [N]
        rets_t = to_t(data["rets"])  # [N]
        old_logps_t = to_t(data["logps"])  # [N]

        idxs = np.arange(N)
        # anneal learning rate linearly
        lr_now = lr * max(0.0, 1.0 - (step_count / float(total_steps)))
        for g in optimizer.param_groups:
            g['lr'] = lr_now

        for epoch in range(update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, N, minibatch_size):
                mb = idxs[start:start+minibatch_size]
                mb_obs = obs_t[mb]
                mb_obs = (mb_obs - obs_mean) / (obs_std + 1e-8)
                mb_obs = torch.nan_to_num(mb_obs)
                mb_obs = torch.clamp(mb_obs, -10.0, 10.0)
                mb_acts = acts_t[mb]
                mb_advs = advs_t[mb]
                mb_rets = rets_t[mb]
                mb_old_logps = old_logps_t[mb]

                mean, std = ac.actor(mb_obs)
                if not torch.isfinite(mean).all():
                    continue
                dist = torch.distributions.Normal(mean, std)
                logps = dist.log_prob(mb_acts).sum(-1)
                ratio = torch.exp(logps - mb_old_logps)
                clip_obj = torch.min(ratio * mb_advs, torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * mb_advs)
                policy_loss = -clip_obj.mean()

                values = ac.value(mb_obs)
                # clipped value function loss
                with torch.no_grad():
                    v_old = ac.value(mb_obs)
                v_pred_clipped = v_old + torch.clamp(values - v_old, -clip_ratio, clip_ratio)
                v_loss_unclipped = F.mse_loss(values, mb_rets)
                v_loss_clipped = F.mse_loss(v_pred_clipped, mb_rets)
                value_loss = torch.max(v_loss_unclipped, v_loss_clipped)

                entropy = dist.entropy().sum(-1).mean()

                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                optimizer.step()
                last_loss = float(loss.item())

                with torch.no_grad():
                    approx_kl = (mb_old_logps - logps).mean().item()
                if approx_kl > 0.03:
                    break
        avg_reward = float(np.mean(data["rews"])) if "rews" in data else float(ep_reward_sum / (ep_len + 1e-6))
        pbar.set_postfix({
            "loss": f"{last_loss:.4f}",
            "reward": f"{avg_reward:.3f}",
            "succ": f"{ep_successes}",
        })

        ep_reward = float(np.sum(data["rews"])) if "rews" in data else float(np.sum(data["rets"]))
        rewards_log.append(ep_reward)
        if save_dir and (ep_reward > best_ret):
            best_ret = ep_reward
            path = os.path.join(save_dir, "best.pt")
            torch.save({
                "model": ac.state_dict(),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "obs_mean": obs_mean.detach().cpu().numpy(),
                "obs_std": obs_std.detach().cpu().numpy(),
                "cfg": {
                    "gamma": gamma, "lam": lam, "clip_ratio": clip_ratio,
                    "vf_coef": vf_coef, "ent_coef": ent_coef, "lr": lr
                }
            }, path)

        # early stop if successes observed in this rollout
        if ep_successes >= 3:
            break

    if save_dir:
        torch.save({
            "model": ac.state_dict(),
            "obs_mean": obs_mean.detach().cpu().numpy(),
            "obs_std": obs_std.detach().cpu().numpy(),
        }, os.path.join(save_dir, "final.pt"))
        np.save(os.path.join(save_dir, "rewards.npy"), np.array(rewards_log, dtype=np.float32))
    env.close()
    return save_dir