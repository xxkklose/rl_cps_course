import os
import gymnasium as gym
import gymnasium_robotics
import manipulator_mujoco
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.buffers import ReplayBuffer
from gymnasium.spaces import Dict as DictSpace
import imageio
import argparse
from tqdm import tqdm

os.environ.setdefault("MUJOCO_GL", "egl")

def make_env(env_id: str = "FetchPickAndPlace-v4"):
    env = gym.make(env_id, render_mode="rgb_array")
    return env

def train_her_sac(
    env_id: str = "FetchPickAndPlace-v4",
    seed: int = 42,
    total_timesteps: int = 1000000,
    buffer_size: int = 1000000,
    learning_rate: float = 1e-3,
    batch_size: int = 512,
    tau: float = 0.05,
    gamma: float = 0.95,
    device: str = "cuda",
):
    env = make_env(env_id)
    env.reset(seed=seed)

    # HER 的配置
    her_kwargs = dict(
        n_sampled_goal=4,              # 每个真实 transition 生成多少条虚拟 transition
        goal_selection_strategy="future",  # 'future' | 'final' | 'episode'
        # 如果 compute_reward 用到了 info，可以打开 copy_info_dict
        copy_info_dict=True,
    )

    is_dict_obs = isinstance(env.observation_space, DictSpace)
    policy_kwargs = dict(net_arch=[256, 256])
    if is_dict_obs:
        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=her_kwargs,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./logs/her_sac",
            seed=seed,
            device=device,
        )
    else:
        model = SAC(
            policy="MlpPolicy",
            env=env,
            replay_buffer_class=ReplayBuffer,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./logs/her_sac",
            seed=seed,
            device=device,
        )

    model.learn(total_timesteps=total_timesteps, log_interval=10, progress_bar=True)
    time_now = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(f"checkpoints/her_sac/{env_id}", exist_ok=True)
    model.save(f"checkpoints/her_sac/{env_id}/{time_now}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=str, default="train")
    parser.add_argument("--video", "-v", type=bool, default=False)
    parser.add_argument("--env", "-e", type=str, default="FetchPickAndPlace-v4")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--total_timesteps", "-t", type=int, default=1000000)
    parser.add_argument("--buffer_size", "-b", type=int, default=1000000)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", "-bs", type=int, default=512)
    parser.add_argument("--tau", "-tau", type=float, default=0.05)
    parser.add_argument("--gamma", "-g", type=float, default=0.95)
    parser.add_argument("--device", "-d", type=str, default="cuda")
    args = parser.parse_args()
    if args.mode == "train":
        train_her_sac(
            env_id=args.env,
            seed=args.seed,
            total_timesteps=args.total_timesteps,
            buffer_size=args.buffer_size,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            device=args.device,
        )
    if args.mode == "test":
        test_her_sac(args.video, env_id=args.env)
