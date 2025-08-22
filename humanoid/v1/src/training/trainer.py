import torch
import numpy as np
import gymnasium as gym
import time
import matplotlib.pyplot as plt
import os
import pickle
from typing import Dict, Any
from tqdm import tqdm

from ..models.agent import CSFAgent
from ..utils.normalizer import StateNormalizer
from ..utils.metrics import compute_state_coverage
from ..utils.csv_logger import CSVLogger, MetricsTracker
from ..evaluation.visualizer import SkillVisualizer
from ..utils.x_pos_wrapper import XPosWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CSFTrainer:
    """Main trainer for CSF algorithm"""

    def __init__(self, config: Dict[str, Any], resume_path:str =None):
        self.config = config

        # Initialize environment with proper Gymnasium v5 API
        env_config = config["environment"]
        env_kwargs = {}

        if env_config.get("render_mode"):
            env_kwargs["render_mode"] = env_config["render_mode"]

        if env_config.get("max_episode_steps"):
            env_kwargs["max_episode_steps"] = env_config["max_episode_steps"]

        env = gym.make(env_config["name"], **env_kwargs)
        self.env = XPosWrapper(env)

        # Get environment dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        print(f"Environment: {env_config['name']}")
        print(f"Observation space: {self.env.observation_space}")
        print(f"Action space: {self.env.action_space}")
        print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        print(f"Device selected: {device}")

        # Initialize agent
        self.agent = CSFAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            skill_dim=config["model"]["skill_dim"],
            repr_dim=config["model"]["repr_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            lr=float(config["training"]["lr"]),
            gamma=config["training"]["gamma"],
            tau=config["training"]["tau"],
            xi=config["training"]["xi"],
            batch_size=config["training"]["batch_size"],
            buffer_size=config["training"]["buffer_size"],
        )

        # State normalizer
        self.normalizer = StateNormalizer(self.state_dim)

        # Visualizer
        self.visualizer = SkillVisualizer()

        # Setup logging
        self.csv_logger = CSVLogger(config["paths"]["log_dir"], "training_log.csv")
        self.metrics_tracker = MetricsTracker(window_size=100)
        
        if resume_path:
            print(f"\n--- Resuming training from checkpoint: {resume_path} ---")
            if not os.path.exists(resume_path):
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

            # Load agent models and optimizers, and get the starting timestep
            start_timestep = self.agent.load_models(resume_path, load_optimizers=True)
            self.timestep = start_timestep
            self.episode = 0 # Episode count is not saved, will restart from 0

            # Load the corresponding normalizer
            normalizer_path = resume_path.replace('.pt', '_normalizer.pkl')
            if os.path.exists(normalizer_path):
                self.normalizer.load(normalizer_path)
                print(f"Loaded normalizer from {normalizer_path}")
            else:
                print(f"Warning: Normalizer file not found at {normalizer_path}. Continuing with a fresh normalizer.")
            
            print(f"Successfully resumed from timestep: {self.timestep}")
            print("Note: The replay buffer will be repopulated from scratch.")
        else:
             # Training state
            self.timestep = 0
            self.episode = 0   

        self.state_positions = []
        self.episode_rewards = []
        self.episode_lengths = []

        # Setup directories
        os.makedirs(config["paths"]["save_dir"], exist_ok=True)
        os.makedirs(config["paths"]["log_dir"], exist_ok=True)

        # Log hyperparameters
        if not resume_path:
            self.csv_logger.log_hyperparameters(config)

    def train(self):
        """Main training loop"""
        total_timesteps = self.config["training"]["total_timesteps"]
        save_interval = self.config["logging"]["save_interval"]
        eval_interval = self.config["logging"]["eval_interval"]
        log_interval = self.config["logging"]["log_interval"]

        print(f"Training CSF on {self.config['environment']['name']}")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Logging to: {self.csv_logger.filepath}")

        start_time = time.time()
        with tqdm(total=total_timesteps, desc="Training", initial=self.timestep) as pbar:
            while self.timestep < total_timesteps:
                self._run_episode()

                # Update progress bar
                pbar.update(self.episode_lengths[-1] if self.episode_lengths else 0)

                # Logging
                if self.timestep % log_interval == 0:
                    self._log_metrics(start_time)

                # Evaluation and visualization
                if self.timestep % eval_interval == 0:
                    self._evaluate()

                # Save models
                if self.timestep % save_interval == 0:
                    self.agent.save_models(
                        self.config["paths"]["save_dir"], self.timestep
                    )
                    # Also save normalizer
                    normalizer_path = os.path.join(
                        self.config["paths"]["save_dir"],
                        f"csf_checkpoint_{self.timestep}_normalizer.pkl",
                    )
                    self.normalizer.save(normalizer_path)

        # Final save
        self.agent.save_models(self.config["paths"]["save_dir"], self.timestep)
        self.normalizer.save(
            os.path.join(self.config["paths"]["save_dir"], "normalizer_final.pkl")
        )
        self._save_final_metrics()

        # Save training summary
        self.csv_logger.save_summary()

        self.env.close()
        print(f"\nTraining completed! Total timesteps: {self.timestep}")
        print(f"Training log saved to: {self.csv_logger.filepath}")
        return self.agent

    def _run_episode(self):
        """Run a single episode with proper Gymnasium v5 API"""
        obs, info = self.env.reset()
        state = obs

        # Sample skill for this episode
        skill = self.agent.sample_skill()

        episode_reward = 0.0
        episode_length = 0
        episode_positions = []

        while True:
            # Normalize state
            self.normalizer.update(state)
            normalized_state = self.normalizer.normalize(state)

            # Convert to tensors
            state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(device)
            skill_tensor = skill.unsqueeze(0).to(device)

            # Sample action
            action = self.agent.policy.sample_action(state_tensor, skill_tensor)
            action_np = action.cpu().numpy().flatten()

            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            # Use wrapper-provided x_pos
            x_position = float(info.get("x_pos", state[0]))
            episode_positions.append(x_position)

            # Normalize next state
            normalized_next_state = self.normalizer.normalize(next_obs)

            # Store transition
            self.agent.replay_buffer.push(
                normalized_state,
                action_np,
                reward,
                normalized_next_state,
                done,
                skill.cpu().numpy(),
            )

            # Update networks
            if len(self.agent.replay_buffer) >= self.agent.batch_size:
                self.agent.update()

            state = next_obs
            episode_reward += reward
            episode_length += 1
            self.timestep += 1

            if done:
                break

        self.state_positions.extend(episode_positions)
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode += 1

    def _log_metrics(self, start_time):
        """Log training metrics to CSV"""
        avg_reward = (
            np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        )
        avg_length = (
            np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
        )
        coverage = compute_state_coverage(
            self.state_positions[-10000:]
        )  # Recent coverage

        elapsed_time = time.time() - start_time

        metrics = {
            "timestep": self.timestep,
            "episode": self.episode,
            "avg_reward_100": avg_reward,
            "avg_episode_length_100": avg_length,
            "state_coverage": coverage,
            "elapsed_time": elapsed_time,
            "buffer_size": len(self.agent.replay_buffer),
            "episodes_per_hour": self.episode / (elapsed_time / 3600)
            if elapsed_time > 0
            else 0,
        }

        if self.episode_rewards:
            metrics["latest_reward"] = self.episode_rewards[-1]
            metrics["latest_episode_length"] = self.episode_lengths[-1]

        if self.agent.metrics["phi_loss"]:
            recent_losses = {
                "phi_loss": np.mean(self.agent.metrics["phi_loss"][-100:]),
                "psi_loss": np.mean(self.agent.metrics["psi_loss"][-100:]),
                "policy_loss": np.mean(self.agent.metrics["policy_loss"][-100:]),
                "positive_term": np.mean(self.agent.metrics["positive_term"][-100:]),
                "negative_term": np.mean(self.agent.metrics["negative_term"][-100:]),
            }
            metrics.update(recent_losses)

        normalizer_stats = self.normalizer.get_stats()
        metrics.update(
            {
                "normalizer_count": normalizer_stats["count"],
                "normalizer_mean_norm": np.linalg.norm(normalizer_stats["mean"]),
                "normalizer_std_norm": np.linalg.norm(normalizer_stats["std"]),
            }
        )

        self.csv_logger.log(metrics, step=self.timestep)
        self.metrics_tracker.update(metrics)

        print(
            f"Timestep: {self.timestep:6d} | "
            f"Episode: {self.episode:4d} | "
            f"Avg Reward: {avg_reward:7.2f} | "
            f"Avg Length: {avg_length:6.1f} | "
            f"Coverage: {coverage:3d} | "
            f"Time: {elapsed_time:6.1f}s"
        )

    def _evaluate(self):
        """Evaluation and visualization"""
        print(f"\n=== Evaluation at timestep {self.timestep} ===")

        eval_metrics = {}

        try:
            # skill_fig = self.visualizer.visualize_skills(
            #     self.agent, self.env, num_skills=8, num_steps=200
            # )

            # skill_fig.savefig(
            #     os.path.join(
            #         self.config["paths"]["log_dir"],
            #         f"skills_{self.timestep}.png",
            #     ),
            #     dpi=150,
            #     bbox_inches="tight",
            # )
            # plt.close(skill_fig)

            # eval_metrics["skill_visualization"] = "success"
            traj_fig = self.visualizer.visualize_skill_trajectories(
                self.agent, self.env, self.normalizer, num_skills=8, num_steps=200
            )
            traj_fig.savefig(
                 os.path.join(
                     self.config["paths"]["log_dir"],
                      f"skill_trajectories_{self.timestep}.png",
                      ),
                  dpi=150,
                 bbox_inches="tight")
            plt.close(traj_fig)

            latent_fig = self.visualizer.visualize_latent_space(
                self.agent, self.env, self.normalizer, num_skills=8, num_steps=200
            )
            latent_fig.savefig(
               os.path.join(
                    self.config["paths"]["log_dir"],
                    f"latent_space_{self.timestep}.png",
                ),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(latent_fig)
 
            eval_metrics["skill_visualization"] = "success"

        except Exception as e:
            print(f"Skill visualization failed: {e}")
            eval_metrics["skill_visualization"] = "failed"

        try:
            coverage_fig = self.visualizer.plot_coverage_history(self.state_positions)

            coverage_fig.savefig(
                os.path.join(
                    self.config["paths"]["log_dir"],
                    f"coverage_{self.timestep}.png",
                ),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(coverage_fig)

            eval_metrics["coverage_plot"] = "success"

        except Exception as e:
            print(f"Coverage plotting failed: {e}")
            eval_metrics["coverage_plot"] = "failed"

        eval_metrics.update(
            {
                "total_coverage": compute_state_coverage(self.state_positions),
                "recent_coverage": compute_state_coverage(self.state_positions[-5000:]),
                "unique_positions": len(set(self.state_positions)),
            }
        )

        eval_log_metrics = {f"eval_{k}": v for k, v in eval_metrics.items()}
        eval_log_metrics["timestep"] = self.timestep
        eval_log_metrics["episode"] = self.episode

        self.csv_logger.log(eval_log_metrics, step=self.timestep)

    def _save_final_metrics(self):
        """Save final training metrics"""
        metrics_path = os.path.join(
            self.config["paths"]["save_dir"], "final_metrics.pkl"
        )
        with open(metrics_path, "wb") as f:
            pickle.dump(
                {
                    "config": self.config,
                    "agent_metrics": dict(self.agent.metrics),
                    "state_positions": self.state_positions,
                    "episode_rewards": self.episode_rewards,
                    "episode_lengths": self.episode_lengths,
                    "normalizer_stats": self.normalizer.get_stats(),
                    "csv_log_data": self.csv_logger.get_logged_data(),
                },
                f,
            )

        print(f"Final metrics saved to: {metrics_path}")
        print(
            f"Final state coverage: {compute_state_coverage(self.state_positions)}"
        )
        print(f"Total episodes: {self.episode}")
        print(f"Average episode reward: {np.mean(self.episode_rewards):.2f}")
        print(f"Average episode length: {np.mean(self.episode_lengths):.1f}")