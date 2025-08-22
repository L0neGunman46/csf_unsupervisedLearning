#!/usr/bin/env python3

import sys
import os
import yaml
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.trainer import CSFTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train CSF on HalfCheetah')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Override save directory')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Override log directory')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Override total timesteps')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint .pt file to resume training from')
    
    args = parser.parse_args()
    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Override settings from command line
    if args.save_dir:
        config['paths']['save_dir'] = args.save_dir
    if args.log_dir:
        config['paths']['log_dir'] = args.log_dir
    if args.timesteps:
        config['training']['total_timesteps'] = args.timesteps
    
    print("Starting CSF training with configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Initialize trainer
    trainer = CSFTrainer(config, resume_path=args.resume)
    
    # Train the agent
    agent = trainer.train()
    
    print("Training completed successfully!")
    print(f"Check logs at: {config['paths']['log_dir']}")
    print(f"Check models at: {config['paths']['save_dir']}")


if __name__ == "__main__":
    main()