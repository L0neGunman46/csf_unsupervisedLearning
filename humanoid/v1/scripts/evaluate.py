#!/usr/bin/env python3

import sys
import os
import yaml
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.evaluator import CSFEvaluator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained CSF agent')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes per skill')
    parser.add_argument('--skills', type=int, default=8,
                        help='Number of skills to evaluate')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum steps per episode')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    
    # Load configuration
    config = load_config(args.config)
    
    print(f"Evaluating CSF agent from: {args.checkpoint}")
    print(f"Episodes per skill: {args.episodes}")
    print(f"Number of skills: {args.skills}")
    
    # Initialize evaluator
    evaluator = CSFEvaluator(config)
    
    # Evaluate the agent
    skill_returns, skill_trajectories = evaluator.evaluate(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        num_skills=args.skills,
        max_steps=args.max_steps
    )
    
    print("Evaluation completed successfully!")
    print("\nResults summary:")
    for i, returns in enumerate(skill_returns):
        print(f"Skill {i+1}: {len(returns)} episodes, "
              f"avg return: {sum(returns)/len(returns):.2f}")


if __name__ == "__main__":
    main()