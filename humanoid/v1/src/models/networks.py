import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P


class StateEncoder(nn.Module):
    """State representation network φ"""

    def __init__(self, state_dim: int, hidden_dim: int = 1024, repr_dim: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, repr_dim)
        )
        P.orthogonal(self.network[-1], "weight")

    def forward(self, state):
        return self.network(state)


class SuccessorFeatures(nn.Module):
    """Successor features network ψ"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        skill_dim: int,
        hidden_dim: int = 1024,
        repr_dim: int = 2,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim + skill_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim),
        )

    def forward(self, state, action, skill):
        x = torch.cat([state, action, skill], dim=-1)
        return self.network(x)


class SkillConditionedPolicy(nn.Module):
    """Skill-conditioned policy π"""

    def __init__(
        self, state_dim: int, action_dim: int, skill_dim: int, hidden_dim: int = 1024
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + skill_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.action_dim = action_dim

    def forward(self, state, skill):
        x = torch.cat([state, skill], dim=-1)
        return torch.tanh(self.network(x))

    def sample_action(self, state, skill, noise_scale=0.1):
        with torch.no_grad():
            action = self.forward(state, skill)
            noise = torch.randn_like(action) * noise_scale
            action = torch.clamp(action + noise, -1, 1)
        return action


class MetaPolicy(nn.Module):
    """High-level policy π_meta(z|s) for HRL"""

    def __init__(self, state_dim: int, skill_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.state_dim = state_dim
        self.skill_dim = skill_dim
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, skill_dim),
        )

    def forward(self, state):
        # Output a vector for the skill
        skill_vec = self.network(state)
        # Normalize to put it on the unit sphere, making it a valid skill
        skill_vec = F.normalize(skill_vec, p=2, dim=-1)
        return skill_vec


class MetaCritic(nn.Module):
    """High-level critic Q_meta(s, z) for HRL"""

    def __init__(self, state_dim: int, skill_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + skill_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, skill):
        x = torch.cat([state, skill], dim=-1)
        return self.network(x)