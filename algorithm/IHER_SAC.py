# IHER_SAC.py
"""
IHER_SAC: subclass of stable_baselines3.SAC that uses AdaptiveHerReplayBuffer by default.
This file provides a drop-in replacement for SB3's SAC with AGR integrated.

Usage:
    from IHER_SAC import IHER_SAC
    model = IHER_SAC("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000_000)
    model.save("iher_sac")
"""

from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.buffers import ReplayBuffer
from gymnasium.spaces import Dict as DictSpace
from typing import Optional, Type
from memory.AGR_HER import AdaptiveHerReplayBuffer


class IHER_SAC(SAC):
    def __init__(
        self,
        policy: str,
        env: GymEnv,
        replay_buffer_class: Optional[Type] = AdaptiveHerReplayBuffer,
        replay_buffer_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        is_dict_obs = isinstance(getattr(env, "observation_space", None), DictSpace)
        if is_dict_obs:
            if replay_buffer_kwargs is None:
                replay_buffer_kwargs = dict(
                    n_sampled_goal=4,
                    goal_selection_strategy="future",
                    copy_info_dict=True,
                    lambda_dist=1.0,
                    lambda_time=0.1,
                    n_candidates=8,
                    use_entropy_aware=False,
                    eags_beta=0.5,
                )
            if replay_buffer_class is None:
                replay_buffer_class = AdaptiveHerReplayBuffer
        else:
            if replay_buffer_class is None or replay_buffer_class is AdaptiveHerReplayBuffer:
                replay_buffer_class = ReplayBuffer
            if replay_buffer_kwargs is None:
                replay_buffer_kwargs = dict()
            if policy == "MultiInputPolicy":
                policy = "MlpPolicy"
        print(f"[IHER_SAC] using policy: {policy}")
        super().__init__(
            policy,
            env,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            **kwargs,
        )

    def _setup_model(self) -> None:
        # call parent setup (creates policy, replay buffer, etc)
        super()._setup_model()

        # If our replay buffer supports set_policy, pass policy to it for entropy-aware sampling
        try:
            if hasattr(self.replay_buffer, "set_policy"):
                # self.policy is a BasePolicy instance in SB3
                self.replay_buffer.set_policy(self.policy)
        except Exception as e:
            # not critical; continue
            print(f"[IHER_SAC] warning: failed to set policy to replay_buffer: {e}")

    @classmethod
    def load(
        cls,
        path: str,
        env: Optional[GymEnv] = None,
        replay_buffer_class: Optional[Type] = None,
        **kwargs,
    ) -> "IHER_SAC":
        """
        Load an IHER_SAC model from file, restoring both model parameters
        and (optionally) the replay buffer if present.

        Parameters
        ----------
        path : str
            Path to the saved model (zip file saved by model.save).
        env : Optional[GymEnv]
            The environment to attach after loading. Required for training.
        replay_buffer_class : type
            Ensures replay buffer loads as AdaptiveHerReplayBuffer.
        kwargs : dict
            Extra arguments passed to SB3's load().

        Returns
        -------
        IHER_SAC
            A fully restored SAC model with improved HER buffer.
        """

        # --- Step 1: create model instance from SB3 loader ---
        model = super(IHER_SAC, cls).load(path, env=env, **kwargs)
        if env is not None:
            is_dict_obs = isinstance(getattr(env, "observation_space", None), DictSpace)
            if replay_buffer_class is None:
                model.replay_buffer_class = AdaptiveHerReplayBuffer if is_dict_obs else ReplayBuffer
            else:
                model.replay_buffer_class = replay_buffer_class

        # --- Step 2: attempt to load replay buffer (.pkl) ---
        # SB3 saves replay buffer under {path}_rb.pkl
        replay_buffer_path = path + "_replay_buffer.pkl"

        try:
            # If the file exists, SB3 will load it
            model.load_replay_buffer(replay_buffer_path)

            # After loading, ensure the buffer still supports AGR
            if hasattr(model.replay_buffer, "set_policy"):
                model.replay_buffer.set_policy(model.policy)

            print(f"[IHER_SAC] Replay buffer successfully loaded from: {replay_buffer_path}")

        except FileNotFoundError:
            print(f"[IHER_SAC] No replay buffer found at: {replay_buffer_path}.")
            print("[IHER_SAC] A new AdaptiveHerReplayBuffer will be created during training.")

        except Exception as e:
            print(f"[IHER_SAC] Failed to load replay buffer: {e}")
            print("[IHER_SAC] A new AdaptiveHerReplayBuffer will be created instead.")

        return model
