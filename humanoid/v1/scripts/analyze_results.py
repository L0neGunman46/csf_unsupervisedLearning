#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def analyze_training_log(log_path: str):
    """Analyze training results from CSV log"""
    
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return
    
    # Load data
    df = pd.read_csv(log_path)
    print(f"Loaded {len(df)} log entries")
    print(f"Columns: {list(df.columns)}")

    # Determine the x-axis column
    # For HRL logs, 'episode' is the primary axis.
    # For main training logs, 'timestep' is the primary axis.
    if 'timestep' in df.columns:
        x_col = 'timestep'
    elif 'episode' in df.columns: # This will catch HRL logs
        x_col = 'episode'
    else:
        print("Error: Could not find a suitable x-axis column ('timestep' or 'episode').")
        return

    print(f"Using '{x_col}' as the x-axis for plots.")
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Average reward over time
    if 'avg_reward_100' in df.columns:
        axes[0, 0].plot(df[x_col], df['avg_reward_100'])
        axes[0, 0].set_title('Average Reward (100 episodes)')
        axes[0, 0].set_xlabel(x_col.capitalize()) # Capitalize for plot label
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        # If avg_reward_100 is not present, check for total_reward (common in HRL)
        if 'total_reward' in df.columns:
            # Calculate a rolling average for total_reward if avg_reward_100 is missing
            rolling_avg_col = f'rolling_avg_total_reward_{min(100, len(df))}'
            df[rolling_avg_col] = df['total_reward'].rolling(window=100, min_periods=1).mean()
            axes[0, 0].plot(df[x_col], df[rolling_avg_col])
            axes[0, 0].set_title(f'Rolling Avg Total Reward (Window={min(100, len(df))})')
            axes[0, 0].set_xlabel(x_col.capitalize())
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].set_title('Reward (Data Not Found)')


    # Plot 2: State coverage over time (This column is only in the main training log)
    if 'state_coverage' in df.columns:
        axes[0, 1].plot(df[x_col], df['state_coverage'])
        axes[0, 1].set_title('State Coverage')
        axes[0, 1].set_xlabel(x_col.capitalize())
        axes[0, 1].set_ylabel('Coverage')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].set_title('State Coverage (N/A for HRL)')
        axes[0, 1].text(0.5, 0.5, "Not logged in this file type", 
                         horizontalalignment='center', verticalalignment='center', 
                         transform=axes[0, 1].transAxes, fontsize=12, color='gray')
    
    # Plot 3: Average episode length over time (This column is only in the main training log)
    if 'avg_episode_length_100' in df.columns:
        axes[0, 2].plot(df[x_col], df['avg_episode_length_100'])
        axes[0, 2].set_title('Average Episode Length (100 episodes)')
        axes[0, 2].set_xlabel(x_col.capitalize())
        axes[0, 2].set_ylabel('Length')
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].set_title('Episode Length (N/A for HRL)')
        axes[0, 2].text(0.5, 0.5, "Not logged in this file type", 
                         horizontalalignment='center', verticalalignment='center', 
                         transform=axes[0, 2].transAxes, fontsize=12, color='gray')
    
    # Plot 4: Losses
    # This might require checking specific HRL losses vs. main training losses
    # For HRL, we only track total_reward, not explicit losses in the log
    loss_columns = [col for col in df.columns if 'loss' in col.lower()]
    if loss_columns:
        for col in loss_columns:
            if col in df.columns:
                axes[1, 0].plot(df[x_col], df[col], label=col, alpha=0.7)
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].set_xlabel(x_col.capitalize())
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        # Only set yscale to log if there are non-zero positive values, otherwise it will crash
        if not df[loss_columns].empty and (df[loss_columns] > 0).any().any():
            axes[1, 0].set_yscale('log')
    else:
        axes[1, 0].set_title('Losses (N/A for HRL meta-controller log)')
        axes[1, 0].text(0.5, 0.5, "Not logged in this file type", 
                         horizontalalignment='center', verticalalignment='center', 
                         transform=axes[1, 0].transAxes, fontsize=12, color='gray')

    # Plot 5: Contrastive terms (These are only in the main training log)
    if 'positive_term' in df.columns and 'negative_term' in df.columns:
        axes[1, 1].plot(df[x_col], df['positive_term'], label='Positive term', alpha=0.7)
        axes[1, 1].plot(df[x_col], df['negative_term'], label='Negative term', alpha=0.7)
        axes[1, 1].set_title('Contrastive Loss Terms')
        axes[1, 1].set_xlabel(x_col.capitalize())
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].set_title('Contrastive Terms (N/A for HRL)')
        axes[1, 1].text(0.5, 0.5, "Not logged in this file type", 
                         horizontalalignment='center', verticalalignment='center', 
                         transform=axes[1, 1].transAxes, fontsize=12, color='gray')
    
    # Plot 6: Training speed (This column is only in the main training log)
    if 'episodes_per_hour' in df.columns:
        axes[1, 2].plot(df[x_col], df['episodes_per_hour'])
        axes[1, 2].set_title('Training Speed')
        axes[1, 2].set_xlabel(x_col.capitalize())
        axes[1, 2].set_ylabel('Episodes/Hour')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].set_title('Training Speed (N/A for HRL)')
        axes[1, 2].text(0.5, 0.5, "Not logged in this file type", 
                         horizontalalignment='center', verticalalignment='center', 
                         transform=axes[1, 2].transAxes, fontsize=12, color='gray')
    
    plt.tight_layout()
    
    # Save plot
    output_path = log_path.replace('.csv', '_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Analysis plot saved to: {output_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n=== Training Summary ===")
    if 'avg_reward_100' in df.columns:
        final_reward = df['avg_reward_100'].iloc[-1] if not df['avg_reward_100'].empty else 0
        max_reward = df['avg_reward_100'].max() if not df['avg_reward_100'].empty else 0
        print(f"Final average reward: {final_reward:.2f}")
        print(f"Maximum average reward: {max_reward:.2f}")
    elif 'total_reward' in df.columns: # For HRL logs
        final_reward = df['total_reward'].iloc[-1] if not df['total_reward'].empty else 0
        max_reward = df['total_reward'].max() if not df['total_reward'].empty else 0
        print(f"Final total reward: {final_reward:.2f}")
        print(f"Maximum total reward: {max_reward:.2f}")
        if 'rolling_avg_total_reward_100' in df.columns:
            final_rolling_avg = df['rolling_avg_total_reward_100'].iloc[-1]
            print(f"Final rolling average total reward: {final_rolling_avg:.2f}")

    if 'state_coverage' in df.columns:
        final_coverage = df['state_coverage'].iloc[-1] if not df['state_coverage'].empty else 0
        max_coverage = df['state_coverage'].max() if not df['state_coverage'].empty else 0
        print(f"Final state coverage: {final_coverage}")
        print(f"Maximum state coverage: {max_coverage}")
    
    if 'elapsed_time' in df.columns:
        total_time = df['elapsed_time'].iloc[-1] if not df['elapsed_time'].empty else 0
        print(f"Total training time: {total_time/3600:.2f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze CSF training results')
    parser.add_argument('--log', type=str, default='logs/training_log.csv',
                        help='Path to training log CSV file')
    
    args = parser.parse_args()
    
    analyze_training_log(args.log)