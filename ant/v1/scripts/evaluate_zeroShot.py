#!/usr/bin/env python3

import sys
import os
import yaml
import argparse
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
import torch.nn.functional as F


# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.agent import CSFAgent
from src.utils.normalizer import StateNormalizer
from src.utils.csv_logger import CSVLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_zeroshot(config: dict, checkpoint_path: str):
    """Evaluate a pre-trained agent on zero-shot goal reaching."""
    
    # --- Setup ---
    env_config = config['environment']
    zeroshot_config = config['zeroshot']
    
    env = gym.make(env_config['name'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Load agent
    agent = CSFAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        **config['model']
    )
    agent.load_models(checkpoint_path)
    agent.phi.eval()
    agent.policy.eval()

    # Load normalizer
    normalizer_path = checkpoint_path.replace('.pt', '_normalizer.pkl')
    if not os.path.exists(normalizer_path):
        print(f"Warning: Normalizer file not found at {normalizer_path}. Using an empty normalizer.")
        normalizer = StateNormalizer(state_dim)
    else:
        normalizer = StateNormalizer(state_dim)
        normalizer.load(normalizer_path)
        print(f"Loaded normalizer from {normalizer_path}")

    # Setup logger
    log_dir = config['paths']['log_dir']
    logger = CSVLogger(log_dir, "zeroshot_results.csv")
    
    print(f"Starting Zero-Shot Goal Reaching evaluation for {checkpoint_path}")
    
    # --- Evaluation Loop ---
    all_staying_times = []
    all_final_distances = []

    for episode in tqdm(range(zeroshot_config['num_episodes']), desc="ZSGR Evaluation"):
        # Sample a goal (x-coordinate)
        goal_x = np.random.uniform(*zeroshot_config['goal_sampling_range'])
        goal_state = env.observation_space.sample() # Sample a full state
        goal_state[0] = goal_x # Set the x-coordinate to our goal

        obs, _ = env.reset()
        state = obs
        
        staying_time = 0
        
        for step in range(zeroshot_config['max_steps']):
            # Normalize states and goal
            norm_state = normalizer.normalize(state)
            norm_goal = normalizer.normalize(goal_state)

            # Infer skill z*
            with torch.no_grad():
                s_tensor = torch.FloatTensor(norm_state).to(device)
                g_tensor = torch.FloatTensor(norm_goal).to(device)
                
                phi_s = agent.phi(s_tensor)
                phi_g = agent.phi(g_tensor)
                
                skill_vec = phi_g - phi_s
                # Normalize to put on unit sphere, handle zero vector case
                if torch.norm(skill_vec) > 1e-6:
                    skill_vec = F.normalize(skill_vec, p=2, dim=-1)
                
            # Execute action
            action = agent.policy.sample_action(s_tensor.unsqueeze(0), skill_vec.unsqueeze(0), noise_scale=0.0)
            
            obs, _, terminated, truncated, _ = env.step(action.cpu().numpy().flatten())
            state = obs

            # Check if goal is reached
            current_dist = np.linalg.norm(state[0] - goal_state[0])
            if current_dist < zeroshot_config['goal_threshold']:
                staying_time += 1

            if terminated or truncated:
                break
        
        final_dist = np.linalg.norm(state[0] - goal_state[0])
        staying_time_fraction = staying_time / (step + 1)
        
        all_staying_times.append(staying_time_fraction)
        all_final_distances.append(final_dist)

        # Log per-episode results
        logger.log({
            'episode': episode,
            'goal_x': goal_x,
            'final_distance': final_dist,
            'staying_time_fraction': staying_time_fraction
        })

    # --- Final Results ---
    avg_staying_time = np.mean(all_staying_times)
    std_staying_time = np.std(all_staying_times)
    avg_final_dist = np.mean(all_final_distances)

    print("\n--- Zero-Shot Goal Reaching Results ---")
    print(f"Average Staying Time Fraction: {avg_staying_time:.3f} ± {std_staying_time:.3f}")
    print(f"Average Final Distance to Goal: {avg_final_dist:.3f}")
    print(f"Results saved to {logger.filepath}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate CSF on Zero-Shot Goal Reaching')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to pre-trained model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate_zeroshot(config, args.checkpoint)