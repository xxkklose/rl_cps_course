# AGR_HER.py
"""
Adaptive HER ReplayBuffer for Stable-Baselines3.

Inherit from stable_baselines3.her.her_replay_buffer.HerReplayBuffer
and override _sample_goals to implement AGR:
    - sample several candidate future achieved_goals for each transition
    - score them by (-lambda_dist * distance - lambda_time * delta_t)
    - optionally add an entropy estimate term if a policy is set
    - softmax-sample a candidate and return it as the relabeled goal

This class exposes set_policy(policy) so the model can provide the policy
for entropy-aware goal sampling.
"""

from typing import Optional
import numpy as np
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy


class AdaptiveHerReplayBuffer(HerReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        env=None,
        device: str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        n_sampled_goal: int = 4,
        goal_selection_strategy: str = "future",
        copy_info_dict: bool = True,
        lambda_dist: float = 1.0,
        lambda_time: float = 0.1,
        n_candidates: int = 8,
        use_entropy_aware: bool = False,
        eags_beta: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            env=env,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy=goal_selection_strategy,
            copy_info_dict=copy_info_dict,
        )

        self.lambda_dist = lambda_dist
        self.lambda_time = lambda_time
        self.n_candidates = n_candidates

        self.use_entropy_aware = use_entropy_aware
        self.eags_beta = eags_beta

        self._policy_for_entropy = None

        self.goal_selection_strategy = GoalSelectionStrategy.FUTURE

    def set_policy(self, policy):
        """
        外部设置 policy（用于熵感知采样）
        """
        self._policy_for_entropy = policy

    def _sample_goals(self, batch_indices: np.ndarray, env_indices: np.ndarray) -> np.ndarray:
        """
        Override HerReplayBuffer._sample_goals to implement AGR.
        """
        ag_shape = self.next_observations["achieved_goal"].shape[2:]
        if len(batch_indices) == 0:
            return np.zeros((0,) + ag_shape)

        new_goals = np.zeros_like(self.next_observations["achieved_goal"][batch_indices, env_indices])

        for i, (buf_idx, env_idx) in enumerate(zip(batch_indices, env_indices)):
            start = int(self.ep_start[buf_idx, env_idx])
            length = int(self.ep_length[buf_idx, env_idx])

            cur_in_ep = (buf_idx - start) % self.buffer_size
            min_future = cur_in_ep + 1
            max_future = length - 1
            if min_future > max_future:
                new_goals[i] = self.next_observations["achieved_goal"][buf_idx, env_idx]
                continue

            # candidate future indices
            n_cand = min(self.n_candidates, max_future - min_future + 1)
            if n_cand <= (max_future - min_future + 1):
                cand_in_ep = np.random.choice(np.arange(min_future, max_future + 1), size=n_cand, replace=False)
            else:
                cand_in_ep = np.random.randint(min_future, max_future + 1, size=n_cand)

            cand_buf_idxs = (cand_in_ep + start) % self.buffer_size

            cur_ag = self.next_observations["achieved_goal"][buf_idx, env_idx]
            cand_ag = self.next_observations["achieved_goal"][cand_buf_idxs, env_idx]

            dists = np.linalg.norm(cand_ag - cur_ag, axis=-1)
            delta_t = cand_in_ep - cur_in_ep
            scores = -self.lambda_dist * dists - self.lambda_time * delta_t.astype(np.float32)

            # optional entropy-aware scoring
            if self.use_entropy_aware and (self._policy_for_entropy is not None):
                try:
                    import torch
                    obs_base = self.next_observations["observation"][buf_idx, env_idx]
                    obs_batch = np.array([np.concatenate([obs_base, g]) for g in cand_ag], dtype=np.float32)
                    obs_t = torch.as_tensor(obs_batch, device=self.device)
                    # 如果 policy 提供 get_entropy_estimate 方法，则使用
                    if hasattr(self._policy_for_entropy, "get_entropy_estimate"):
                        ent = self._policy_for_entropy.get_entropy_estimate(obs_t)
                        ent = ent.cpu().numpy().reshape(-1)
                        scores += self.eags_beta * ent
                except Exception:
                    pass  # 忽略错误，fallback 到普通 AGR

            # softmax 选择
            max_s = np.max(scores)
            exp_s = np.exp(scores - max_s)
            weights = exp_s / (np.sum(exp_s) + 1e-8)
            chosen = np.random.choice(np.arange(len(weights)), p=weights)
            new_goals[i] = self.next_observations["achieved_goal"][cand_buf_idxs[chosen], env_idx]

        return new_goals
