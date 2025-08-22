# File: src/evaluation/visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Dict
from sklearn.decomposition import PCA

from ..utils.metrics import compute_state_coverage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SkillVisualizer:
    """Visualization utilities for CSF"""

    def visualize_skill_trajectories(
        self, agent, env, normalizer, num_skills: int = 8, num_steps: int = 200
    ):
        """
        Visualize learned skills by overlaying their trajectories in state space.
        This is similar to Figure 12 in the METRA paper.
        """
        print("Generating skill trajectory plot...")
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, num_skills))

        agent.policy.eval()

        for i in range(num_skills):
            skill = agent.sample_skill().to(device)
            obs, info = env.reset()
            state = obs

            # Store x and y positions for 2D envs, or x vs time for 1D
            positions = []
            for _ in range(num_steps):
                state_tensor = torch.from_numpy(normalizer.normalize(state)).float().unsqueeze(0).to(device)
                skill_tensor = skill.unsqueeze(0)

                action = agent.policy.sample_action(
                    state_tensor, skill_tensor, noise_scale=0.0
                )
                action = action.cpu().numpy().flatten()

                next_obs, _, terminated, truncated, info = env.step(action)
                
                # Use the x_pos from our wrapper
                x_pos = info.get("x_pos", 0.0)
                # For Ant/Quadruped/Humanoid, y is often in obs[1] or from qpos[1]
                try:
                    y_pos = env.unwrapped.data.qpos[1]
                except:
                    y_pos = None # Fallback for envs like HalfCheetah
                
                positions.append((x_pos, y_pos))
                state = next_obs
                if terminated or truncated:
                    break
            
            positions = np.array(positions)
            x_coords = positions[:, 0]
            y_coords = positions[:, 1]
            if y_coords.size > 0 and y_coords[0] is not None:  # 2D plot (Ant, Quadruped, etc.)
                ax.plot(x_coords, y_coords, color=colors[i], alpha=0.7, linewidth=2, label=f'Skill {i+1}')
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
            else: # 1D plot (HalfCheetah)
                ax.plot(np.arange(len(x_coords)), x_coords, color=colors[i], alpha=0.7, linewidth=2, label=f'Skill {i+1}')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('X Position')

        ax.set_title(f'State Space Trajectories for {num_skills} Skills')
        ax.grid(True, alpha=0.3)
        # ax.legend() # Legend can get crowded, optional
        
        plt.tight_layout()
        return fig

    def visualize_latent_space(
        self, agent, env, normalizer, num_skills: int = 8, num_steps: int = 200
    ):
        """
        Visualize the latent space representations phi(s) for different skills.
        Uses PCA to project to 2D if repr_dim > 2.
        """
        print("Generating latent space plot...")
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, num_skills))

        agent.phi.eval()
        
        all_latents = []
        all_skill_indices = []

        for i in range(num_skills):
            skill = agent.sample_skill().to(device)
            obs, info = env.reset()
            state = obs
            
            for _ in range(num_steps):
                state_tensor = torch.from_numpy(normalizer.normalize(state)).float().unsqueeze(0).to(device)
                
                with torch.no_grad():
                    latent_rep = agent.phi(state_tensor).cpu().numpy().flatten()
                
                all_latents.append(latent_rep)
                all_skill_indices.append(i)

                skill_tensor = skill.unsqueeze(0)
                action = agent.policy.sample_action(
                    state_tensor, skill_tensor, noise_scale=0.0
                ).cpu().numpy().flatten()
                
                state, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
        
        all_latents = np.array(all_latents)
        all_skill_indices = np.array(all_skill_indices)

        # If latent space is > 2D, use PCA to visualize
        if all_latents.shape[1] > 2:
            print(f"Latent space dim is {all_latents.shape[1]}, using PCA to project to 2D.")
            pca = PCA(n_components=2)
            latents_2d = pca.fit_transform(all_latents)
        else:
            latents_2d = all_latents

        scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1], c=all_skill_indices, cmap='viridis', alpha=0.5, s=10)
        
        ax.set_title('Latent Space phi(s) Colored by Skill')
        ax.set_xlabel('Latent Dimension 1 (or PC1)')
        ax.set_ylabel('Latent Dimension 2 (or PC2)')
        ax.grid(True, alpha=0.3)
        
        # Create a legend
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Skill {i+1}',
                                  markerfacecolor=colors[i], markersize=10) for i in range(num_skills)]
        ax.legend(handles=legend_handles, title="Skills")

        plt.tight_layout()
        return fig

    def plot_coverage_history(self, state_positions: List[float]):
        """Plot state coverage over time"""
        plt.figure(figsize=(10, 6))
        coverage_history = []
        window_size = 1000
        
        for i in range(0, len(state_positions), window_size):
            window_positions = state_positions[i:i+window_size]
            coverage = compute_state_coverage(window_positions) 
            coverage_history.append(coverage)
        
        plt.plot(coverage_history)
        plt.title('State Coverage Over Time')
        plt.xlabel('Time (x1000 steps)')
        plt.ylabel('Coverage (unique grid cells)')
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()

    def visualize_evaluation_results(
        self, 
        skill_returns: List[List[float]], 
        skill_trajectories: List[List[Dict]], 
        num_skills: int
    ):
        """Visualize evaluation results from the main evaluator script"""
        # This function remains the same as before
        plt.figure(figsize=(12, 8))
        
        # Plot returns per skill
        plt.subplot(2, 2, 1)
        returns_data = [returns for returns in skill_returns]
        plt.boxplot(returns_data, labels=[f'Skill {i+1}' for i in range(num_skills)])
        plt.title('Returns per Skill')
        plt.ylabel('Episode Return')
        plt.xticks(rotation=45)
        
        # Plot x-position trajectories
        plt.subplot(2, 2, 2)
        colors = plt.cm.tab10(np.linspace(0, 1, num_skills))
        
        for skill_idx in range(min(num_skills, 8)):
            traj = skill_trajectories[skill_idx][0]
            x_positions = [step['x_pos'] for step in traj]
            plt.plot(x_positions, color=colors[skill_idx], 
                    label=f'Skill {skill_idx+1}', alpha=0.7, linewidth=2)
        
        plt.title('X-Position Trajectories')
        plt.xlabel('Time Steps')
        plt.ylabel('X Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot coverage comparison
        plt.subplot(2, 2, 3)
        skill_coverages = []
        for skill_idx in range(num_skills):
            positions = []
            for traj in skill_trajectories[skill_idx]:
                positions.extend([step['x_pos'] for step in traj])
            coverage = compute_state_coverage(positions) 
            skill_coverages.append(coverage)
        
        plt.bar(range(1, num_skills+1), skill_coverages)
        plt.title('Coverage per Skill')
        plt.xlabel('Skill')
        plt.ylabel('Coverage')
        
        # Plot skill diversity (pairwise distances)
        plt.subplot(2, 2, 4)
        skill_distances = np.zeros((num_skills, num_skills))
        
        for i in range(num_skills):
            for j in range(num_skills):
                if i != j:
                    traj_i = [step['x_pos'] for step in skill_trajectories[i][0]]
                    traj_j = [step['x_pos'] for step in skill_trajectories[j][0]]
                    
                    if len(traj_i) > 0 and len(traj_j) > 0:
                        skill_distances[i, j] = abs(traj_i[-1] - traj_j[-1])
        
        sns.heatmap(skill_distances, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Skill Distance Matrix')
        plt.xlabel('Skill')
        plt.ylabel('Skill')
        
        plt.tight_layout()
        plt.savefig('csf_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return plt.gcf()