import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import defaultdict
import os

from .networks import StateEncoder, SuccessorFeatures, SkillConditionedPolicy
from ..utils.replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CSFAgent:
    """Contrastive Successor Features Agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        skill_dim: int = 2,
        repr_dim: int = 2,
        hidden_dim: int = 1024,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        xi: float = 5.0,
        batch_size: int = 256,
        buffer_size: int = 1000000,
        beta_ib: float = 1e-3,
        phi_l2_reg: float = 1e-3,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.skill_dim = skill_dim
        self.repr_dim = repr_dim
        self.gamma = gamma
        self.tau = tau
        self.xi = xi
        self.batch_size = batch_size
        self.beta_ib = beta_ib
        self.phi_l2_reg = phi_l2_reg

        # Networks
        self.phi = StateEncoder(state_dim, hidden_dim, repr_dim).to(device)
        self.psi = SuccessorFeatures(
            state_dim, action_dim, skill_dim, hidden_dim, repr_dim
        ).to(device)
        self.psi_target = SuccessorFeatures(
            state_dim, action_dim, skill_dim, hidden_dim, repr_dim
        ).to(device)
        self.policy = SkillConditionedPolicy(
            state_dim, action_dim, skill_dim, hidden_dim
        ).to(device)

        # Project skill into representation space for contrastive objective
        assert (
         repr_dim == skill_dim
        ), f"CSF requires repr_dim == skill_dim, got {repr_dim} vs {skill_dim}"

        # Copy parameters to target network
        self.psi_target.load_state_dict(self.psi.state_dict())

        # Optimizers
        self.phi_optimizer = optim.Adam(self.phi.parameters(), lr=lr)
        self.psi_optimizer = optim.Adam(self.psi.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        # also optimize skill_to_repr
        self.phi_optimizer.add_param_group({"params": self.skill_to_repr.parameters()})

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Metrics
        self.metrics = defaultdict(list)

    def sample_skill(self):
        """Sample skill from uniform distribution on unit hypersphere"""
        skill = torch.randn(self.skill_dim, device=device, dtype=torch.float32)
        norm = torch.norm(skill)
        skill = skill / (norm + 1e-8)
        return skill

    def contrastive_loss(self, states, next_states, skills):
        """
        CSF contrastive lower bound on Iβ(S,S';Z) (Eq. 10 variant).
        states, next_states: [B, state_dim] (normalized)
        skills: [B, d] unit-norm, where d = repr_dim = skill_dim
        L = - E[(phi(s')-phi(s))·z] + alpha * E[log (1/(B-1) * sum_{z'≠z} exp((phi(s')-phi(s))·z'))]
       """
        B = states.shape[0]
        phi_s = self.phi(states)        # [B, d]
        phi_s_next = self.phi(next_states)  # [B, d]
        diff = phi_s_next - phi_s       # [B, d]

        # Positive term: align diff with its own z
        pos = (diff * skills).sum(dim=1)      # [B]
        pos_term = pos.mean()

        # In-batch negatives: logits = diff @ skills^T, mask out diagonal
        logits = diff @ skills.T              # [B, B]
        mask = torch.eye(B, device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(mask, float("-inf"))
        # log-mean-exp over negatives (exclude positive)
        lse = torch.logsumexp(logits, dim=1) - torch.log(
            torch.tensor(float(B - 1), device=logits.device, dtype=logits.dtype)
        )
        neg_term = lse.mean()

        alpha = self.xi  # scale negative only
        loss = -pos_term + alpha * neg_term

        # small L2 on representations (optional, as in your code)
        if self.phi_l2_reg > 0:
            loss = loss + self.phi_l2_reg * (phi_s.pow(2).sum(dim=1).mean())

        return loss, pos_term.item(), neg_term.item()

    def successor_features_loss(
        self, states, actions, next_states, skills, next_actions
    ):
        """Compute successor features loss"""
        # Current successor features
        psi_current = self.psi(states, actions, skills)

        # Target successor features
        phi_s_next = self.phi(next_states)
        phi_s = self.phi(states)
        repr_diff = phi_s_next - phi_s

        with torch.no_grad():
            psi_next = self.psi_target(next_states, next_actions, skills)
            target = repr_diff + self.gamma * psi_next

        loss = F.mse_loss(psi_current, target)
        return loss

    def policy_loss(self, states, actions, skills):
        """Compute policy loss using successor features"""
        psi_values = self.psi(states, actions, skills)
        loss = -torch.mean(torch.sum(psi_values * skills, dim=1))
        return loss

    def update(self):
        """Update all networks"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            skills,
        ) = self.replay_buffer.sample(self.batch_size)

        # Sample next actions for successor features update
        next_actions = self.policy.sample_action(
            next_states, skills, noise_scale=0.1
        )

        # Update state encoder (φ) + skill projector
        phi_loss, pos_term, neg_term = self.contrastive_loss(
            states, next_states, skills
        )
        self.phi_optimizer.zero_grad()
        phi_loss.backward()
        self.phi_optimizer.step()

        # Update successor features (ψ)
        psi_loss = self.successor_features_loss(
            states, actions, next_states, skills, next_actions
        )
        self.psi_optimizer.zero_grad()
        psi_loss.backward()
        self.psi_optimizer.step()

        # Update policy (π)
        policy_actions = self.policy(states, skills)
        pol_loss = self.policy_loss(states, policy_actions, skills)
        self.policy_optimizer.zero_grad()
        pol_loss.backward()
        self.policy_optimizer.step()

        # Soft update target network
        for target_param, param in zip(
            self.psi_target.parameters(), self.psi.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        # Store metrics
        self.metrics["phi_loss"].append(phi_loss.item())
        self.metrics["psi_loss"].append(psi_loss.item())
        self.metrics["policy_loss"].append(pol_loss.item())
        self.metrics["positive_term"].append(pos_term)
        self.metrics["negative_term"].append(neg_term)

    def save_models(self, save_dir: str, iteration: int):
        """Save model checkpoints"""
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            "iteration": iteration,
            "phi_state_dict": self.phi.state_dict(),
            "psi_state_dict": self.psi.state_dict(),
            "psi_target_state_dict": self.psi_target.state_dict(),
            "policy_state_dict": self.policy.state_dict(),
            "phi_optimizer": self.phi_optimizer.state_dict(),
            "psi_optimizer": self.psi_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
        }

        torch.save(
            checkpoint, os.path.join(save_dir, f"csf_checkpoint_{iteration}.pt")
        )
        print(f"Models saved at iteration {iteration}")

    def load_models(self, checkpoint_path: str, load_optimizers: bool = True):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.phi.load_state_dict(checkpoint["phi_state_dict"])
        self.psi.load_state_dict(checkpoint["psi_state_dict"])
        self.psi_target.load_state_dict(checkpoint["psi_target_state_dict"])
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
       

        if load_optimizers:
            self.phi_optimizer.load_state_dict(checkpoint["phi_optimizer"])
            self.psi_optimizer.load_state_dict(checkpoint["psi_optimizer"])
            self.policy_optimizer.load_state_dict(
                checkpoint["policy_optimizer"]
            )

        print(f"Models loaded from {checkpoint_path}")
        return checkpoint.get("iteration", -1)