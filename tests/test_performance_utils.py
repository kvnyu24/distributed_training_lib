import os
import tempfile
import time
import shutil
import numpy as np
from mpi4py import MPI
from distributed_training_lib import (
    PerformanceMonitor, 
    Timer, 
    measure_allreduce_bandwidth,
    ModelParallelConfig,
    ParallelTrainer
)

def test_timer():
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Test basic timer functionality
    timer = Timer("test_timer")
    timer.start()
    time.sleep(0.1)  # Sleep for 100ms
    elapsed = timer.stop()
    
    if rank == 0:
        print(f"Timer elapsed time: {elapsed:.4f}s")
        assert elapsed >= 0.1, f"Timer elapsed time should be >= 0.1s, got {elapsed:.4f}s"
        assert elapsed < 0.2, f"Timer elapsed time should be < 0.2s, got {elapsed:.4f}s"
        print("Timer test passed!")

def test_performance_monitor():
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Create a temporary directory for logs
    if rank == 0:
        temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {temp_dir}")
    else:
        temp_dir = None
    
    # Broadcast the directory path to all processes
    temp_dir = comm.bcast(temp_dir, root=0)
    
    # Create performance monitor
    monitor = PerformanceMonitor(
        comm=comm,
        rank=rank,
        log_dir=os.path.join(temp_dir, "logs")
    )
    
    # Simulate a few training steps
    for i in range(3):
        # Start step
        monitor.step()
        
        # Simulate computation
        monitor.start_timer("compute")
        time.sleep(0.1)  # Sleep for 100ms
        monitor.stop_timer("compute")
        
        # Simulate communication
        monitor.start_timer("comm")
        comm.Barrier()  # Simple communication
        monitor.stop_timer("comm")
        
        # Add a custom metric
        monitor.add_metric("loss", 0.1 / (i + 1))
        
        # Increment a counter
        monitor.increment_counter("num_samples", 10)
    
    # Log metrics
    monitor.log_metrics()
    
    # Get metrics
    metrics = monitor.get_metrics()
    
    if rank == 0:
        # Verify metrics
        print(f"Performance metrics: {metrics}")
        assert metrics["steps"] == 3, f"Expected 3 steps, got {metrics['steps']}"
        assert metrics["total_compute_time"] > 0, f"Expected compute time > 0, got {metrics['total_compute_time']:.4f}s"
        assert metrics["total_comm_time"] > 0, f"Expected comm time > 0, got {metrics['total_comm_time']:.4f}s"
        assert metrics["counters"]["num_samples"] == 30, f"Expected 30 samples, got {metrics['counters']['num_samples']}"
        assert len(metrics) > 5, f"Expected at least 5 metrics, got {len(metrics)}"
        
        print("Performance monitor test passed!")
    
    # Clean up
    if rank == 0:
        shutil.rmtree(temp_dir)
        print(f"Removed temporary directory: {temp_dir}")

def test_bandwidth_measurement():
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Measure allreduce bandwidth with small data
    mean_bw, std_bw = measure_allreduce_bandwidth(
        comm=comm,
        size_mb=0.1,  # 100KB
        trials=3
    )
    
    if rank == 0:
        print(f"Allreduce bandwidth: {mean_bw:.2f} ± {std_bw:.2f} MB/s")
        assert mean_bw > 0, f"Expected positive bandwidth, got {mean_bw:.2f} MB/s"
        print("Bandwidth measurement test passed!")

if __name__ == "__main__":
    print("\n=== Timer Test ===")
    test_timer()
    
    print("\n=== Performance Monitor Test ===")
    test_performance_monitor()
    
    print("\n=== Bandwidth Measurement Test ===")
    test_bandwidth_measurement() 