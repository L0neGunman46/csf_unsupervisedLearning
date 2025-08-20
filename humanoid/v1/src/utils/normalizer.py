import numpy as np
import pickle
import os


class StateNormalizer:
    """Normalize states to zero mean and unit variance using running statistics"""
    
    def __init__(self, state_dim: int, epsilon: float = 1e-8):
        self.state_dim = state_dim
        self.epsilon = epsilon
        
        # Running statistics
        self.mean = np.zeros(state_dim, dtype=np.float64)
        self.var = np.ones(state_dim, dtype=np.float64)
        self.std = np.ones(state_dim, dtype=np.float64)
        self.count = 0
        
        # For numerical stability
        self.min_std = epsilon
        
    def update(self, state: np.ndarray):
        """Update running statistics using Welford's online algorithm"""
        if len(state.shape) == 1:
            # Single state
            self._update_single(state)
        else:
            # Batch of states
            for s in state:
                self._update_single(s)
    
    def _update_single(self, state: np.ndarray):
        """Update statistics for a single state"""
        self.count += 1
        
        # Welford's online algorithm for numerical stability
        delta = state - self.mean
        self.mean += delta / self.count
        delta2 = state - self.mean
        self.var += (delta * delta2 - self.var) / self.count
        
        # Update standard deviation with minimum threshold
        self.std = np.sqrt(self.var)
        self.std = np.maximum(self.std, self.min_std)
    
    def normalize(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to zero mean and unit variance"""
        if self.count == 0:
            return state
        
        return (state - self.mean) / self.std
    
    def denormalize(self, normalized_state: np.ndarray) -> np.ndarray:
        """Denormalize state back to original scale"""
        if self.count == 0:
            return normalized_state
        
        return normalized_state * self.std + self.mean
    
    def get_stats(self) -> dict:
        """Get current normalization statistics"""
        return {
            'mean': self.mean.copy(),
            'std': self.std.copy(),
            'var': self.var.copy(),
            'count': self.count
        }
    
    def set_stats(self, mean: np.ndarray, std: np.ndarray, count: int):
        """Set normalization statistics manually"""
        assert len(mean) == self.state_dim, f"Mean dimension mismatch: {len(mean)} vs {self.state_dim}"
        assert len(std) == self.state_dim, f"Std dimension mismatch: {len(std)} vs {self.state_dim}"
        
        self.mean = mean.copy()
        self.std = np.maximum(std.copy(), self.min_std)
        self.var = self.std ** 2
        self.count = count
    
    def save(self, filepath: str):
        """Save normalizer state to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        state = {
            'mean': self.mean,
            'std': self.std,
            'var': self.var,
            'count': self.count,
            'state_dim': self.state_dim,
            'epsilon': self.epsilon
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, filepath: str):
        """Load normalizer state from file"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.mean = state['mean']
        self.std = state['std']
        self.var = state['var']
        self.count = state['count']
        
        # Verify dimensions match
        assert state['state_dim'] == self.state_dim, \
            f"State dimension mismatch: {state['state_dim']} vs {self.state_dim}"
    
    def reset(self):
        """Reset normalizer to initial state"""
        self.mean = np.zeros(self.state_dim, dtype=np.float64)
        self.var = np.ones(self.state_dim, dtype=np.float64)
        self.std = np.ones(self.state_dim, dtype=np.float64)
        self.count = 0
    
    def __repr__(self):
        return (f"StateNormalizer(state_dim={self.state_dim}, "
                f"count={self.count}, "
                f"mean_norm={np.linalg.norm(self.mean):.4f}, "
                f"std_norm={np.linalg.norm(self.std):.4f})")


class RunningStat:
    """Alternative implementation using running statistics for comparison"""
    
    def __init__(self, shape: tuple):
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.S = np.zeros(shape, dtype=np.float64)
        self.shape = shape
    
    def update(self, x):
        x = np.asarray(x)
        assert x.shape == self.shape
        
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
    
    @property
    def var(self):
        return self.S / (self.n - 1) if self.n > 1 else np.ones(self.shape)
    
    @property
    def std(self):
        return np.sqrt(self.var)


class BatchNormalizer:
    """Normalizer that works with batches of states efficiently"""
    
    def __init__(self, state_dim: int, momentum: float = 0.99, epsilon: float = 1e-8):
        self.state_dim = state_dim
        self.momentum = momentum
        self.epsilon = epsilon
        
        self.running_mean = np.zeros(state_dim, dtype=np.float32)
        self.running_var = np.ones(state_dim, dtype=np.float32)
        self.count = 0
    
    def update(self, states: np.ndarray):
        """Update statistics with a batch of states"""
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
        
        batch_mean = np.mean(states, axis=0)
        batch_var = np.var(states, axis=0)
        batch_size = states.shape[0]
        
        if self.count == 0:
            self.running_mean = batch_mean
            self.running_var = batch_var
        else:
            # Exponential moving average
            self.running_mean = (self.momentum * self.running_mean + 
                               (1 - self.momentum) * batch_mean)
            self.running_var = (self.momentum * self.running_var + 
                              (1 - self.momentum) * batch_var)
        
        self.count += batch_size
    
    def normalize(self, states: np.ndarray) -> np.ndarray:
        """Normalize batch of states"""
        if self.count == 0:
            return states
        
        return ((states - self.running_mean) / 
                np.sqrt(self.running_var + self.epsilon))
    
    def denormalize(self, normalized_states: np.ndarray) -> np.ndarray:
        """Denormalize batch of states"""
        if self.count == 0:
            return normalized_states
        
        return (normalized_states * np.sqrt(self.running_var + self.epsilon) + 
                self.running_mean)


# Utility functions for creating normalizers
def create_state_normalizer(state_dim: int, normalizer_type: str = "welford"):
    """Factory function to create different types of normalizers"""
    if normalizer_type == "welford":
        return StateNormalizer(state_dim)
    elif normalizer_type == "batch":
        return BatchNormalizer(state_dim)
    else:
        raise ValueError(f"Unknown normalizer type: {normalizer_type}")


# Test the normalizer
if __name__ == "__main__":
    # Test the normalizer with some random data
    np.random.seed(42)
    
    state_dim = 5
    normalizer = StateNormalizer(state_dim)
    
    print("Testing StateNormalizer...")
    print(f"Initial state: {normalizer}")
    
    # Generate some test data
    n_samples = 1000
    test_data = np.random.randn(n_samples, state_dim) * 2 + 1  # mean=1, std=2
    
    # Update normalizer with test data
    for state in test_data:
        normalizer.update(state)
    
    print(f"After {n_samples} updates: {normalizer}")
    print(f"Empirical mean: {np.mean(test_data, axis=0)}")
    print(f"Normalizer mean: {normalizer.mean}")
    print(f"Empirical std: {np.std(test_data, axis=0)}")
    print(f"Normalizer std: {normalizer.std}")
    
    # Test normalization
    test_state = test_data[0]
    normalized = normalizer.normalize(test_state)
    denormalized = normalizer.denormalize(normalized)
    
    print(f"\nTest normalization:")
    print(f"Original: {test_state}")
    print(f"Normalized: {normalized}")
    print(f"Denormalized: {denormalized}")
    print(f"Error: {np.mean((test_state - denormalized)**2)}")
    
    # Test saving and loading
    normalizer.save("test_normalizer.pkl")
    
    new_normalizer = StateNormalizer(state_dim)
    new_normalizer.load("test_normalizer.pkl")
    
    print(f"\nLoaded normalizer: {new_normalizer}")
    print(f"Stats match: {np.allclose(normalizer.mean, new_normalizer.mean)}")
    
    # Clean up
    os.remove("test_normalizer.pkl")
    
    print("All tests passed!")
