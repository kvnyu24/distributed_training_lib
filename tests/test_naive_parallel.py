import numpy as np
from mpi4py import MPI
from distributed_training_lib import ModelParallelConfig, ParallelTrainer
from distributed_training_lib.parallel.parallel_ops import ParallelOperations

def test_naive_forward():
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Create configuration
    config = ModelParallelConfig(
        mp_size=2,
        dp_size=1,
        is_megatron=False,
        in_dim=4,
        out_dim=6
    )
    
    # Create sample data
    if rank == 0:
        x = np.array([[1., 2.], [3., 4.]])  # 2x2 matrix
    else:
        x = np.array([[5., 6.], [7., 8.]])  # 2x2 matrix
    
    # Initialize trainer to get MPI communicators
    trainer = ParallelTrainer(config)
    mp_comm = trainer.mp_comm
    
    # Test forward operations
    collected_input = ParallelOperations.naive_forward_input(x, mp_comm, config.mp_size)
    collected_output = ParallelOperations.naive_forward_output(x, mp_comm, config.mp_size)
    
    # Verify results
    if rank == 0:
        print(f"Rank {rank} - Input shape: {collected_input.shape}")
        print(f"Rank {rank} - Input:\n{collected_input}")
        print(f"Rank {rank} - Output shape: {collected_output.shape}")
        print(f"Rank {rank} - Output:\n{collected_output}")
        
        # Expected shapes
        assert collected_input.shape == (2, 4), f"Expected shape (2, 4), got {collected_input.shape}"
        assert collected_output.shape == (2, 4), f"Expected shape (2, 4), got {collected_output.shape}"
        
        # Expected values for input
        expected_input = np.array([[1., 2., 5., 6.], [3., 4., 7., 8.]])
        np.testing.assert_array_almost_equal(collected_input, expected_input)

def test_naive_backward():
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Create configuration
    config = ModelParallelConfig(
        mp_size=2,
        dp_size=1,
        is_megatron=False,
        in_dim=4,
        out_dim=6
    )
    
    # Create sample gradients
    grad = np.array([[0.1, 0.2], [0.3, 0.4]])  # Same gradient for both processes
    
    # Initialize trainer to get MPI communicators
    trainer = ParallelTrainer(config)
    mp_comm = trainer.mp_comm
    
    # Test backward operations
    output_grad = ParallelOperations.naive_backward_output(grad, rank, config.mp_size)
    grad_x = ParallelOperations.naive_backward_x(grad, mp_comm, config.mp_size)
    
    # Verify results
    if rank == 0:
        print(f"Rank {rank} - Output gradient shape: {output_grad.shape}")
        print(f"Rank {rank} - Output gradient:\n{output_grad}")
        print(f"Rank {rank} - Input gradient shape: {grad_x.shape}")
        print(f"Rank {rank} - Input gradient:\n{grad_x}")
        
        # Expected shapes
        assert output_grad.shape == (2, 1), f"Expected shape (2, 1), got {output_grad.shape}"
        assert grad_x.shape == (2, 2), f"Expected shape (2, 2), got {grad_x.shape}"
        
        # Expected values
        expected_grad_x = grad * 2  # Since we sum gradients from both processes
        np.testing.assert_array_almost_equal(grad_x, expected_grad_x)

if __name__ == "__main__":
    test_naive_forward()
    test_naive_backward() 