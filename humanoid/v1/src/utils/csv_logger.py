import csv
import os
import time
from typing import Dict, Any, Optional
from collections import defaultdict
import numpy as np


class CSVLogger:
    """Simple CSV logger for training metrics"""
    
    def __init__(self, log_dir: str, filename: str = "training_log.csv"):
        self.log_dir = log_dir
        self.filename = filename
        self.filepath = os.path.join(log_dir, filename)
        
        # Create directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Track if header has been written
        self.header_written = False
        self.fieldnames = None
        
        # Store all logged data for analysis
        self.logged_data = []
        
        print(f"CSV Logger initialized: {self.filepath}")
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to CSV file"""
        # Add timestamp and step if provided
        log_entry = {
            'timestamp': time.time(),
            'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        if step is not None:
            log_entry['step'] = step
        
        # Add metrics
        log_entry.update(metrics)
        
        # Store for later analysis
        self.logged_data.append(log_entry.copy())
        
        # Write to CSV
        self._write_to_csv(log_entry)
    
    def _write_to_csv(self, log_entry: Dict[str, Any]):
        """Write a single log entry to CSV"""
        # Determine fieldnames from first entry
        if not self.header_written:
            self.fieldnames = list(log_entry.keys())
            
            # Write header
            with open(self.filepath, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
            
            self.header_written = True
        
        # Append data
        with open(self.filepath, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            
            # Only write fields that exist in header
            filtered_entry = {k: v for k, v in log_entry.items() if k in self.fieldnames}
            writer.writerow(filtered_entry)
    
    def log_hyperparameters(self, config: Dict[str, Any]):
        """Log hyperparameters to a separate file"""
        config_filepath = os.path.join(self.log_dir, "hyperparameters.csv")
        
        # Flatten nested config
        flat_config = self._flatten_dict(config)
        
        with open(config_filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['parameter', 'value'])
            for key, value in flat_config.items():
                writer.writerow([key, value])
        
        print(f"Hyperparameters saved to: {config_filepath}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def get_logged_data(self):
        """Get all logged data"""
        return self.logged_data
    
    def save_summary(self):
        """Save training summary statistics"""
        if not self.logged_data:
            return
        
        summary_filepath = os.path.join(self.log_dir, "training_summary.csv")
        
        # Calculate summary statistics for numeric columns
        numeric_columns = []
        for key in self.logged_data[0].keys():
            if key not in ['timestamp', 'datetime', 'step']:
                try:
                    # Test if column is numeric
                    values = [entry[key] for entry in self.logged_data if key in entry and entry[key] is not None]
                    if values:
                        float(values[0])
                        numeric_columns.append(key)
                except (ValueError, TypeError):
                    continue
        
        # Calculate statistics
        summary_data = []
        for col in numeric_columns:
            values = [float(entry[col]) for entry in self.logged_data 
                     if col in entry and entry[col] is not None]
            
            if values:
                summary_data.append({
                    'metric': col,
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'final': values[-1] if values else None
                })
        
        # Write summary
        with open(summary_filepath, 'w', newline='') as csvfile:
            if summary_data:
                fieldnames = ['metric', 'count', 'mean', 'std', 'min', 'max', 'final']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_data)
        
        print(f"Training summary saved to: {summary_filepath}")


class MetricsTracker:
    """Helper class to track and compute running averages of metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(list)
        
    def update(self, metrics: Dict[str, float]):
        """Update metrics with new values"""
        for key, value in metrics.items():
            if value is not None:
                self.metrics[key].append(float(value))
                
                # Keep only recent values
                if len(self.metrics[key]) > self.window_size:
                    self.metrics[key] = self.metrics[key][-self.window_size:]
    
    def get_averages(self) -> Dict[str, float]:
        """Get running averages of all metrics"""
        averages = {}
        for key, values in self.metrics.items():
            if values:
                averages[f"avg_{key}"] = np.mean(values)
                averages[f"std_{key}"] = np.std(values) if len(values) > 1 else 0.0
        return averages
    
    def get_latest(self) -> Dict[str, float]:
        """Get latest values of all metrics"""
        latest = {}
        for key, values in self.metrics.items():
            if values:
                latest[f"latest_{key}"] = values[-1]
        return latest
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    import shutil
    
    # Create temporary directory for testing
    test_dir = tempfile.mkdtemp()
    
    try:
        # Test CSV logger
        logger = CSVLogger(test_dir, "test_log.csv")
        
        # Log some sample data
        for i in range(10):
            metrics = {
                'episode': i,
                'reward': np.random.normal(100, 10),
                'loss': np.random.exponential(0.1),
                'coverage': np.random.randint(10, 100)
            }
            logger.log(metrics, step=i * 100)
        
        # Log hyperparameters
        config = {
            'model': {
                'lr': 0.001,
                'hidden_dim': 256
            },
            'training': {
                'batch_size': 32,
                'epochs': 100
            }
        }
        logger.log_hyperparameters(config)
        
        # Save summary
        logger.save_summary()
        
        # Test metrics tracker
        tracker = MetricsTracker(window_size=5)
        for i in range(10):
            tracker.update({
                'reward': np.random.normal(100, 10),
                'loss': np.random.exponential(0.1)
            })
        
        print("Running averages:", tracker.get_averages())
        print("Latest values:", tracker.get_latest())
        
        print(f"Test files created in: {test_dir}")
        print("Files:", os.listdir(test_dir))
        
    finally:
        # Clean up
        shutil.rmtree(test_dir)
        print("Test completed successfully!")