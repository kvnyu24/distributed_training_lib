import time
import numpy as np
from mpi4py import MPI
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import json
import os

class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self, name: str = ""):
        """Initialize timer.
        
        Args:
            name: Name of the timer
        """
        self.name = name
        self.start_time = None
        self.elapsed = 0.0
        
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        
    def stop(self) -> float:
        """Stop the timer and return elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        
        self.elapsed = time.time() - self.start_time
        self.start_time = None
        return self.elapsed
    
    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.elapsed = 0.0

class PerformanceMonitor:
    """Class to monitor performance metrics in distributed training."""
    
    def __init__(
        self,
        comm: MPI.Comm,
        rank: int,
        log_dir: Optional[str] = None
    ):
        """Initialize the performance monitor.
        
        Args:
            comm: MPI communicator
            rank: Process rank
            log_dir: Directory for logging performance metrics
        """
        self.comm = comm
        self.rank = rank
        self.log_dir = log_dir
        
        # Create log directory if provided
        if log_dir and self.rank == 0:
            os.makedirs(log_dir, exist_ok=True)
        
        # Initialize timers and counters
        self.timers = {}
        self.counters = defaultdict(int)
        self.metrics = defaultdict(list)
        
        # Global performance metrics
        self.steps = 0
        self.total_comm_time = 0.0
        self.total_compute_time = 0.0
        
    def create_timer(self, name: str) -> Timer:
        """Create a named timer.
        
        Args:
            name: Name of the timer
            
        Returns:
            Timer object
        """
        timer = Timer(name)
        self.timers[name] = timer
        return timer
    
    def start_timer(self, name: str):
        """Start a named timer.
        
        Args:
            name: Name of the timer
        """
        if name not in self.timers:
            self.timers[name] = Timer(name)
        self.timers[name].start()
    
    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed time.
        
        Args:
            name: Name of the timer
            
        Returns:
            Elapsed time in seconds
        """
        if name not in self.timers:
            return 0.0
        
        elapsed = self.timers[name].stop()
        
        # Update global metrics
        if "comm" in name.lower():
            self.total_comm_time += elapsed
        elif "compute" in name.lower():
            self.total_compute_time += elapsed
        
        # Save the metric
        self.metrics[name].append(elapsed)
        
        return elapsed
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a named counter.
        
        Args:
            name: Name of the counter
            value: Value to increment by
        """
        self.counters[name] += value
    
    def add_metric(self, name: str, value: float):
        """Add a metric value.
        
        Args:
            name: Name of the metric
            value: Metric value
        """
        self.metrics[name].append(value)
    
    def step(self):
        """Increment the step counter."""
        self.steps += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            'steps': self.steps,
            'total_comm_time': self.total_comm_time,
            'total_compute_time': self.total_compute_time,
            'avg_comm_time': self.total_comm_time / max(1, self.steps),
            'avg_compute_time': self.total_compute_time / max(1, self.steps),
            'counters': dict(self.counters)
        }
        
        # Calculate statistics for each metric
        for name, values in self.metrics.items():
            if values:
                metrics[f"{name}_avg"] = np.mean(values)
                metrics[f"{name}_min"] = np.min(values)
                metrics[f"{name}_max"] = np.max(values)
                metrics[f"{name}_std"] = np.std(values)
        
        return metrics
    
    def log_metrics(self, step: Optional[int] = None):
        """Log current performance metrics.
        
        Args:
            step: Current step number (if None, uses internal counter)
        """
        if self.rank != 0:
            return
        
        step = step if step is not None else self.steps
        metrics = self.get_metrics()
        
        # Print metrics
        print(f"Step {step} Performance Metrics:")
        print(f"  Total Communication Time: {metrics['total_comm_time']:.4f}s")
        print(f"  Total Computation Time: {metrics['total_compute_time']:.4f}s")
        print(f"  Comm/Compute Ratio: {metrics['total_comm_time'] / max(1e-6, metrics['total_compute_time']):.4f}")
        
        # Log to file if log_dir is provided
        if self.log_dir:
            log_file = os.path.join(self.log_dir, f"performance_metrics_{step}.json")
            with open(log_file, 'w') as f:
                json.dump(metrics, f, indent=2)
    
    def reset(self):
        """Reset all timers and counters."""
        for timer in self.timers.values():
            timer.reset()
            
        self.counters = defaultdict(int)
        self.metrics = defaultdict(list)
        self.steps = 0
        self.total_comm_time = 0.0
        self.total_compute_time = 0.0

def measure_allreduce_bandwidth(
    comm: MPI.Comm, 
    size_mb: float = 10.0, 
    trials: int = 5
) -> Tuple[float, float]:
    """Measure allreduce bandwidth in MB/s.
    
    Args:
        comm: MPI communicator
        size_mb: Size of data in MB
        trials: Number of trials
        
    Returns:
        Tuple of (bandwidth_mean, bandwidth_std) in MB/s
    """
    size_bytes = int(size_mb * 1024 * 1024)
    num_elements = size_bytes // 8  # 8 bytes per float64
    
    bandwidths = []
    
    for _ in range(trials):
        # Generate random data
        data = np.random.random(num_elements).astype(np.float64)
        result = np.zeros_like(data)
        
        # Synchronize before timing
        comm.Barrier()
        
        # Time allreduce operation
        start_time = time.time()
        comm.Allreduce(data, result, op=MPI.SUM)
        elapsed = time.time() - start_time
        
        # Calculate bandwidth
        if elapsed > 0:
            bandwidth = size_mb / elapsed  # MB/s
            bandwidths.append(bandwidth)
    
    return np.mean(bandwidths), np.std(bandwidths) 