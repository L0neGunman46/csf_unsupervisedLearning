#!/usr/bin/env python3
"""
scripts/train_hrl.py
Hierarchical fine-tuning after CSF pre-training.
Freezes φ and ψ, trains a high-level policy π_meta(z|s) with SAC.
"""

import argparse
import time
from typing import Dict, Any

import gymnasium as gym
import numpy as np
import torch
import yaml
import os
import sys
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
from src.models.agent import CSFAgent
from src.models.networks import MetaPolicy, MetaCritic
from src.utils.normalizer import StateNormalizer
from src.utils.replay_buffer import ReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAC:
    """Minimal SAC for the high-level policy."""

    def __init__(
        self,
        policy: MetaPolicy,
        critic: MetaCritic,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
    ):
        self.policy = policy
        self.critic = critic
        self.target_critic = MetaCritic(policy.state_dim, policy.skill_dim).to(device)
        self.target_critic.load_state_dict(critic.state_dict())

        self.policy_opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.policy(state)

    def update(self, buffer: ReplayBuffer, batch_size: int):
        if len(buffer) < batch_size:
            return
        states, skills, rewards, next_states, dones = buffer.sample_hrl(batch_size)

        dones_f = dones.float()

        # critic loss
        with torch.no_grad():
            next_q = self.target_critic(next_states, skills)
            target_q = rewards + self.gamma * next_q * (1 - dones_f)
        current_q = self.critic(states, skills)
        critic_loss = ((current_q - target_q) ** 2).mean()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # policy loss
        policy_skill = self.policy(states)
        q_new = self.critic(states, policy_skill)
        policy_loss = -q_new.mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        # soft update target
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


def train_hrl(config: Dict[str, Any], checkpoint_path: str):
    env_name = config["environment"]["name"]
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    skill_dim = config["model"]["skill_dim"]

    # 1. load pre-trained low-level agent
    agent = CSFAgent(
        state_dim=state_dim,
        action_dim=env.action_space.shape[0],
        skill_dim=skill_dim,
        repr_dim=config["model"]["repr_dim"],
        hidden_dim=config["model"]["hidden_dim"],
    )
    agent.load_models(checkpoint_path, load_optimizers=False)

    normalizer_path = checkpoint_path.replace(".pt", "_normalizer.pkl")
    normalizer = StateNormalizer(state_dim)
    normalizer.load(normalizer_path)

    # 2. high-level networks
    meta_policy = MetaPolicy(state_dim, skill_dim).to(device)
    meta_critic = MetaCritic(state_dim, skill_dim).to(device)
    sac = SAC(meta_policy, meta_critic, lr=config["hrl"]["meta_policy_lr"])

    replay = ReplayBuffer(config["hrl"]["hrl_buffer_size"])
    option_len = config["hrl"]["option_timesteps"]

    total_episodes = config["hrl"]["total_episodes"]
    batch_size = config["hrl"]["hrl_batch_size"]

    start_time = time.time()
    for episode in range(total_episodes):
        obs, _ = env.reset()
        state = obs
        episode_return = 0.0
        
        done = False
        
        while not done:
            prev_state = state

            # High-level decision
            s_tensor = (
                torch.FloatTensor(normalizer.normalize(state))
                .unsqueeze(0)
                .to(device)
            )
            skill_vec = sac.select_action(s_tensor).cpu().numpy().flatten()

            # low-level rollout
            option_return = 0.0
            for _ in range(option_len):
                st = (
                    torch.FloatTensor(normalizer.normalize(state))
                    .unsqueeze(0)
                    .to(device)
                )
                sk = torch.FloatTensor(skill_vec).unsqueeze(0).to(device)
                action = (
                    agent.policy.sample_action(st, sk, noise_scale=0.0)
                    .cpu()
                    .numpy()
                    .flatten()
                )

                state, reward, term, trunc, _ = env.step(action)
                option_return += reward
                if term or trunc:
                    break

            done = term or trunc
            # Push the correct (s, z, r, s', d) transition
            replay.push_hrl(prev_state, skill_vec, option_return, state, done)
            episode_return += option_return
            if done:
                break

        sac.update(replay, batch_size)

        if episode % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f"Episode {episode:5d}/{total_episodes} | "
                f"Return: {episode_return:8.2f} | "
                f"Time: {elapsed:6.1f}s"
            )

    env.close()
    print("Hierarchical fine-tuning finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--checkpoint", required=True, help="CSF checkpoint (.pt)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_hrl(cfg, args.checkpoint)