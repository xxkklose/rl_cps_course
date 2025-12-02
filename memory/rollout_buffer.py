import numpy as np

class RolloutBuffer:
    def __init__(self, size, obs_dim, act_dim, gamma=0.99, lam=0.95):
        self.size = size
        self.gamma = gamma
        self.lam = lam
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((size, act_dim), dtype=np.float32)
        self.logps = np.zeros(size, dtype=np.float32)
        self.rews = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.vals = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.path_start_idx = 0
        self.adv = np.zeros(size, dtype=np.float32)
        self.ret = np.zeros(size, dtype=np.float32)

    def store(self, obs, act, logp, rew, done, val):
        assert self.ptr < self.size
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.logps[self.ptr] = logp
        self.rews[self.ptr] = rew
        self.dones[self.ptr] = done
        self.vals[self.ptr] = val
        self.ptr += 1

    def finish_path(self, last_val=0.0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rews[path_slice], last_val)
        vals = np.append(self.vals[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = np.zeros_like(deltas)
        a = 0.0
        for t in reversed(range(len(deltas))):
            a = deltas[t] + self.gamma * self.lam * a
            adv[t] = a
        self.adv[path_slice] = adv
        self.ret[path_slice] = adv + self.vals[path_slice]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.size
        adv = self.adv
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        data = {
            "obs": self.obs.copy(),
            "acts": self.acts.copy(),
            "logps": self.logps.copy(),
            "rews": self.rews.copy(),
            "rets": self.ret.copy(),
            "advs": adv.copy(),
            "vals": self.vals.copy(),
        }
        self.ptr = 0
        self.path_start_idx = 0
        return data
