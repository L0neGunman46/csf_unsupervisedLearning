import torch
import numpy as np
import gymnasium as gym
from typing import List, Tuple, Dict
import torch.nn.functional as F

from ..models.agent import CSFAgent
from ..utils.normalizer import StateNormalizer
from .visualizer import SkillVisualizer
from ..utils.x_pos_wrapper import XPosWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CSFEvaluator:
    """Evaluator for trained CSF agent"""

    def __init__(self, config: dict):
        self.config = config
        self.visualizer = SkillVisualizer()

    def evaluate(
        self,
        checkpoint_path: str,
        num_episodes: int = 10,
        num_skills: int = 8,
        max_steps: int = 1000,
    ) -> Tuple[List[List[float]], List[List[Dict]]]:
        """Evaluate trained CSF agent and compute zero-shot success rate."""

        # 1. Environment setup
        env = gym.make(self.config["environment"]["name"], render_mode="rgb_array")
        env = XPosWrapper(env)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # 2. Load pre-trained agent
        agent = CSFAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            skill_dim=self.config["model"]["skill_dim"],
            repr_dim=self.config["model"]["repr_dim"],
            hidden_dim=self.config["model"]["hidden_dim"],
        )
        iteration = agent.load_models(checkpoint_path)

        normalizer = StateNormalizer(state_dim)
        norm_path = checkpoint_path.replace(".pt", "_normalizer.pkl")
        normalizer.load(norm_path)

        print(f"Evaluating CSF agent from iteration {iteration}")

        # 3. Evaluate each skill
        skill_returns: List[List[float]] = []
        skill_trajectories: List[List[Dict]] = []

        for skill_idx in range(num_skills):
            skill = agent.sample_skill()
            episode_returns: List[float] = []
            episode_trajectories: List[Dict] = []

            for _ in range(num_episodes):
                obs, _ = env.reset()
                state = obs
                trajectory: List[Dict] = []
                total_return = 0.0

                for _ in range(max_steps):
                    normalized_state = normalizer.normalize(state)
                    state_tensor = (
                        torch.from_numpy(normalized_state)
                        .float()
                        .unsqueeze(0)
                        .to(device)
                    )
                    skill_tensor = skill.float().unsqueeze(0).to(device)

                    action = agent.policy.sample_action(
                        state_tensor, skill_tensor, noise_scale=0.0
                    )
                    action_np = action.cpu().numpy().flatten()

                    next_obs, reward, term, trunc, info = env.step(action_np)
                    x_pos = info.get("x_pos", float(next_obs[0]))
                    trajectory.append({"x_pos": x_pos})
                    total_return += reward
                    state = next_obs

                    if term or trunc:
                        break

                episode_returns.append(total_return)
                episode_trajectories.append(trajectory)

            skill_returns.append(episode_returns)
            skill_trajectories.append(episode_trajectories)

            avg_return = np.mean(episode_returns)
            std_return = np.std(episode_returns)
            print(
                f"Skill {skill_idx+1}: Return = {avg_return:.2f} ± {std_return:.2f}"
            )

        # 4. Zero-shot goal-reaching evaluation (infer skill instead of sampling)
        np.random.seed(0)
        goals = np.random.uniform(-100, 100, size=50).tolist()
        zero_shot_trajs: List[List[Dict]] = []

        agent.phi.eval()
        agent.policy.eval()

        for goal in goals:
            traj: List[Dict] = []
            obs, _ = env.reset()
            state = obs

            goal_state = np.array(state, dtype=np.float32).copy()
            goal_state[0] = goal

            for _ in range(max_steps):
                norm_state = normalizer.normalize(state)
                norm_goal = normalizer.normalize(goal_state)

                s_tensor = torch.from_numpy(norm_state).float().to(device)
                g_tensor = torch.from_numpy(norm_goal).float().to(device)

                with torch.no_grad():
                    phi_s = agent.phi(s_tensor)
                    phi_g = agent.phi(g_tensor)
                    skill_vec = phi_g - phi_s
                    if torch.norm(skill_vec) > 1e-6:
                        skill_vec = F.normalize(skill_vec, p=2, dim=-1)

                act = (
                    agent.policy.sample_action(
                        s_tensor.unsqueeze(0),
                        skill_vec.unsqueeze(0),
                        noise_scale=0.0,
                    )
                    .cpu()
                    .numpy()
                    .flatten()
                )

                obs, _, term, trunc, info = env.step(act)
                x_pos = info.get("x_pos", float(obs[0]))
                traj.append({"x_pos": x_pos})

                state = obs
                if term or trunc:
                    break

            zero_shot_trajs.append(traj)

        success_rate = self.compute_success_rate(zero_shot_trajs, goals)
        print(f"Zero-shot success rate: {success_rate:.3f}")

        # 5. Visualize results
        self.visualizer.visualize_evaluation_results(
            skill_returns, skill_trajectories, num_skills
        )

        env.close()
        return skill_returns, skill_trajectories

    @staticmethod
    def compute_success_rate(
        trajectories: List[List[Dict]], goals: List[float], threshold: float = 0.5
    ) -> float:
        """Compute fraction of trajectories that reach within threshold of each goal."""
        successes = 0
        for traj, goal in zip(trajectories, goals):
            if not traj:
                continue
            final_pos = traj[-1]["x_pos"]
            if abs(final_pos - goal) < threshold:
                successes += 1
        return successes / len(goals) if goals else 0.0