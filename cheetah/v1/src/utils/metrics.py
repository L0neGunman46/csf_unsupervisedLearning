# File: src/utils/metrics.py

import numpy as np
from typing import List


def compute_state_coverage(positions: List[float], grid_size: int = 50, 
                         x_range: tuple = (-50, 50)) -> int:
    """Compute state coverage by counting unique grid cells visited
    
    Args:
        positions: A list of x-coordinates (float values).
        grid_size: The number of bins to discretize the x_range into.
        x_range: A tuple (x_min, x_max) defining the range for discretization.
    
    Returns:
        The number of unique grid cells visited.
    """
    x_min, x_max = x_range
    grid = set()
    
    for x_pos in positions: # 'pos' is now 'x_pos' directly, which is the float value
        # Discretize position to grid
        x_idx = int((x_pos - x_min) / (x_max - x_min) * grid_size)
        x_idx = max(0, min(grid_size - 1, x_idx)) # Clamp to valid grid indices
        grid.add(x_idx)
    
    return len(grid)


def compute_skill_diversity(skill_trajectories: List[List[dict]]) -> float:
    """Compute skill diversity based on trajectory differences"""
    if len(skill_trajectories) < 2:
        return 0.0
    
    distances = []
    for i in range(len(skill_trajectories)):
        for j in range(i + 1, len(skill_trajectories)):
            # Get final positions
            traj_i = skill_trajectories[i][0] if skill_trajectories[i] else []
            traj_j = skill_trajectories[j][0] if skill_trajectories[j] else []
            
            if traj_i and traj_j:
                pos_i = traj_i[-1]['x_pos'] if 'x_pos' in traj_i[-1] else 0
                pos_j = traj_j[-1]['x_pos'] if 'x_pos' in traj_j[-1] else 0
                distances.append(abs(pos_i - pos_j))
    
    return np.mean(distances) if distances else 0.0