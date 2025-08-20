# File: scripts/visualize_skills.py

import sys
import os
import yaml
import argparse
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.agent import CSFAgent
from src.utils.normalizer import StateNormalizer
from src.evaluation.visualizer import SkillVisualizer
from src.utils.x_pos_wrapper import XPosWrapper

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Visualize learned CSF skills')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file used for training')
    parser.add_argument('--skills', type=int, default=16,
                        help='Number of skills to visualize')
    parser.add_argument('--steps', type=int, default=500,
                        help='Maximum steps per trajectory')
    
    args = parser.parse_args()
    
    # --- Setup ---
    config = load_config(args.config)
    env_config = config['environment']
    
    env = gym.make(env_config['name'])
    env = XPosWrapper(env) # Use wrapper to get reliable x_pos
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # --- Load Agent and Normalizer ---
    agent = CSFAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        **config['model']
    )
    agent.load_models(args.checkpoint, load_optimizers=False)
    
    normalizer = StateNormalizer(state_dim)
    normalizer_path = args.checkpoint.replace('.pt', '_normalizer.pkl')
    if os.path.exists(normalizer_path):
        normalizer.load(normalizer_path)
        print(f"Loaded normalizer from {normalizer_path}")
    else:
        print("Warning: Normalizer not found. Using un-normalized states.")

    # --- Generate Visualizations ---
    visualizer = SkillVisualizer()
    
    # 1. State Space Trajectory Plot
    traj_fig = visualizer.visualize_skill_trajectories(
        agent, env, normalizer, num_skills=args.skills, num_steps=args.steps
    )
    traj_output_path = os.path.join(config['paths']['log_dir'], "skill_trajectories.png")
    traj_fig.savefig(traj_output_path, dpi=300)
    print(f"Saved trajectory plot to {traj_output_path}")
    
    # 2. Latent Space Plot
    latent_fig = visualizer.visualize_latent_space(
        agent, env, normalizer, num_skills=args.skills, num_steps=args.steps
    )
    latent_output_path = os.path.join(config['paths']['log_dir'], "latent_space.png")
    latent_fig.savefig(latent_output_path, dpi=300)
    print(f"Saved latent space plot to {latent_output_path}")

    # Show plots
    plt.show()
    
    env.close()

if __name__ == "__main__":
    main()