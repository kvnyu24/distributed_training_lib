import numpy as np
from mpi4py import MPI
from distributed_training_lib import ModelParallelConfig, ParallelTrainer, GradientManager

def test_gradient_averaging():
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Create configuration with data parallelism
    config = ModelParallelConfig(
        mp_size=1,
        dp_size=2,
        is_megatron=False,
        in_dim=4,
        out_dim=2
    )
    
    # Initialize trainer to get MPI communicators
    trainer = ParallelTrainer(config)
    dp_comm = trainer.dp_comm
    
    # Create different gradients for each process
    grad_value = float(rank + 1)  # Process 0: 1.0, Process 1: 2.0
    gradients = np.ones((2, 4)) * grad_value
    
    # Test gradient averaging
    averaged_gradients = GradientManager.average_gradients(gradients, dp_comm)
    
    # Expected average: (1.0 + 2.0) / 2 = 1.5
    expected_avg = 1.5
    
    if rank == 0:
        print(f"Rank {rank} - Original gradients:\n{gradients}")
        print(f"Rank {rank} - Averaged gradients:\n{averaged_gradients}")
        assert np.allclose(averaged_gradients, np.ones_like(averaged_gradients) * expected_avg), \
            f"Expected average {expected_avg}, got {averaged_gradients[0, 0]}"
        print("Gradient averaging test passed!")

def test_gradient_collection():
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Create configuration with data parallelism
    config = ModelParallelConfig(
        mp_size=1,
        dp_size=2,
        is_megatron=False,
        in_dim=4,
        out_dim=2
    )
    
    # Initialize trainer to get MPI communicators
    trainer = ParallelTrainer(config)
    dp_comm = trainer.dp_comm
    
    # Create different gradients for each process
    grad_value = float(rank + 1)
    gradients = np.ones((2, 2)) * grad_value
    
    # Test gradient collection
    collected_gradients = GradientManager.collect_gradients(gradients, dp_comm)
    
    if rank == 0:
        print(f"Rank {rank} - Original gradients:\n{gradients}")
        print(f"Rank {rank} - Collected gradients:")
        for i, grad in enumerate(collected_gradients):
            print(f"  Process {i}:\n{grad}")
        
        assert len(collected_gradients) == 2, f"Expected 2 gradients, got {len(collected_gradients)}"
        assert np.allclose(collected_gradients[0], np.ones_like(collected_gradients[0]) * 1.0)
        assert np.allclose(collected_gradients[1], np.ones_like(collected_gradients[1]) * 2.0)
        print("Gradient collection test passed!")

def test_reduce_gradients_dict():
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Create configuration with data parallelism
    config = ModelParallelConfig(
        mp_size=1,
        dp_size=2,
        is_megatron=False,
        in_dim=4,
        out_dim=2
    )
    
    # Initialize trainer to get MPI communicators
    trainer = ParallelTrainer(config)
    dp_comm = trainer.dp_comm
    
    # Create different gradients for each process
    grad_value = float(rank + 1)
    gradients_dict = {
        "layer1": np.ones((2, 2)) * grad_value,
        "layer2": np.ones((3, 3)) * grad_value * 2
    }
    
    # Test gradient reduction with averaging
    reduced_gradients = GradientManager.reduce_gradients(gradients_dict, dp_comm, average=True)
    
    # Test gradient reduction without averaging
    summed_gradients = GradientManager.reduce_gradients(gradients_dict, dp_comm, average=False)
    
    if rank == 0:
        print(f"Rank {rank} - Original gradients:")
        for name, grad in gradients_dict.items():
            print(f"  {name}:\n{grad}")
        
        print(f"Rank {rank} - Averaged gradients:")
        for name, grad in reduced_gradients.items():
            print(f"  {name}:\n{grad}")
            # Check if averaging worked correctly
            expected_avg = (1.0 + 2.0) / 2 if name == "layer1" else (2.0 + 4.0) / 2
            assert np.allclose(grad, np.ones_like(grad) * expected_avg), \
                f"Expected average {expected_avg} for {name}, got {grad[0, 0]}"
        
        print(f"Rank {rank} - Summed gradients:")
        for name, grad in summed_gradients.items():
            print(f"  {name}:\n{grad}")
            # Check if summing worked correctly
            expected_sum = 1.0 + 2.0 if name == "layer1" else 2.0 + 4.0
            assert np.allclose(grad, np.ones_like(grad) * expected_sum), \
                f"Expected sum {expected_sum} for {name}, got {grad[0, 0]}"
        
        print("Gradient dictionary reduction test passed!")

if __name__ == "__main__":
    test_gradient_averaging()
    test_gradient_collection()
    test_reduce_gradients_dict() 