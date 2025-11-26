import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        mean = self.net(x)
        std = self.log_std.exp()
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
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim)

    def act(self, obs):
        mean, std = self.actor(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, dist

    def value(self, obs):
        return self.critic(obs)
