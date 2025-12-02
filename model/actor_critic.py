import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_low=None, act_high=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))
        if act_low is not None and act_high is not None:
            low = torch.as_tensor(act_low, dtype=torch.float32)
            high = torch.as_tensor(act_high, dtype=torch.float32)
            self.register_buffer('act_bias', (high + low) / 2.0)
            self.register_buffer('act_scale', (high - low) / 2.0)
        else:
            self.register_buffer('act_bias', torch.zeros(act_dim))
            self.register_buffer('act_scale', torch.ones(act_dim))

    def forward(self, x):
        raw = self.net(x)
        mean = self.act_bias + self.act_scale * torch.tanh(raw)
        std = self.log_std.exp().clamp(1e-1, 1.0)
        return mean, std

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_low=None, act_high=None):
        super().__init__()
        self.actor = Actor(obs_dim, act_dim, act_low, act_high)
        self.critic = Critic(obs_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def act(self, obs):
        mean, std = self.actor(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, dist

    def value(self, obs):
        return self.critic(obs)
