import torch
import numpy as np
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, skill):
        # Ensure all pushed items are NumPy arrays for consistent stacking
         # Ensure all pushed items are NumPy arrays for consistent stacking
        # Ensure skill is unit-norm (defensive; should already be)
        skill = np.asarray(skill, dtype=np.float32)
        n = np.linalg.norm(skill) + 1e-8
        skill = skill / n
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            np.float32(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
            skill
        ))
    
    def push_hrl(self, state:np.ndarray, skill:np.ndarray, reward, next_state:np.ndarray, done):
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            np.asarray(skill, dtype=np.float32),
            np.float32(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done)
        ))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # Unzip the batch
        states, actions, rewards, next_states, dones, skills = zip(*batch)
        
        # Stack into NumPy arrays first, then convert to tensors
        states = torch.from_numpy(np.stack(states)).to(device)
        actions = torch.from_numpy(np.stack(actions)).to(device)
        rewards = torch.from_numpy(np.stack(rewards)).to(device)
        next_states = torch.from_numpy(np.stack(next_states)).to(device)
        dones = torch.from_numpy(np.stack(dones)).to(device)
        skills = torch.from_numpy(np.stack(skills)).to(device)
        
        return states, actions, rewards, next_states, dones, skills
    
    def sample_hrl(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # Unzip the batch
        states, skills, rewards, next_states, dones = zip(*batch)
        
        # Stack into NumPy arrays first, then convert to tensors
        states = torch.from_numpy(np.stack(states)).to(device)
        skills = torch.from_numpy(np.stack(skills)).to(device)
        rewards = torch.from_numpy(np.stack(rewards)).to(device)
        next_states = torch.from_numpy(np.stack(next_states)).to(device)
        dones = torch.from_numpy(np.stack(dones)).to(device)
        
        return states, skills, rewards, next_states, dones
        
    def __len__(self):
        return len(self.buffer)
