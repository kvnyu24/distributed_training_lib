import numpy as np
from mpi4py import MPI
from distributed_training_lib import ModelParallelConfig, ParallelTrainer
from distributed_training_lib.parallel.parallel_ops import ParallelOperations

def test_megatron_forward():
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Create configuration
    config = ModelParallelConfig(
        mp_size=2,
        dp_size=1,
        is_megatron=True,
        in_dim=4,
        out_dim=6
    )
    
    # Create sample data - in Megatron, input is split across processes
    if rank == 0:
        x = np.array([[1., 2.], [3., 4.]])  # First half of input
    else:
        x = np.array([[5., 6.], [7., 8.]])  # Second half of input
    
    # Initialize trainer to get MPI communicators
    trainer = ParallelTrainer(config)
    mp_comm = trainer.mp_comm
    
    # Test forward operations
    collected_input = ParallelOperations.megatron_forward_input(x, mp_comm, config.mp_size)
    out = np.array([[0.1, 0.2], [0.3, 0.4]])  # Sample output
    collected_output = ParallelOperations.megatron_forward_output(out, mp_comm, config.mp_size)
    
    # Verify results
    if rank == 0:
        print(f"Rank {rank} - Input shape: {collected_input.shape}")
        print(f"Rank {rank} - Input:\n{collected_input}")
        print(f"Rank {rank} - Output shape: {collected_output.shape}")
        print(f"Rank {rank} - Output:\n{collected_output}")
        
        # In Megatron, input stays the same (no gathering)
        assert collected_input.shape == (2, 2), f"Expected shape (2, 2), got {collected_input.shape}"
        np.testing.assert_array_almost_equal(collected_input, x)
        
        # Output is summed across processes
        expected_output = out * 2  # Since both processes contribute
        np.testing.assert_array_almost_equal(collected_output, expected_output)

def test_megatron_backward():
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Create configuration
    config = ModelParallelConfig(
        mp_size=2,
        dp_size=1,
        is_megatron=True,
        in_dim=4,
        out_dim=6
    )
    
    # Create sample gradients
    grad = np.array([[0.1, 0.2], [0.3, 0.4]])  # Same gradient for both processes
    
    # Initialize trainer to get MPI communicators
    trainer = ParallelTrainer(config)
    mp_comm = trainer.mp_comm
    
    # Test backward operations
    output_grad = ParallelOperations.megatron_backward_output(grad, rank, config.mp_size)
    grad_x = ParallelOperations.megatron_backward_x(grad, mp_comm, config.mp_size)
    
    # Verify results
    if rank == 0:
        print(f"Rank {rank} - Output gradient shape: {output_grad.shape}")
        print(f"Rank {rank} - Output gradient:\n{output_grad}")
        print(f"Rank {rank} - Input gradient shape: {grad_x.shape}")
        print(f"Rank {rank} - Input gradient:\n{grad_x}")
        
        # In Megatron, output gradient stays the same
        assert output_grad.shape == (2, 2), f"Expected shape (2, 2), got {output_grad.shape}"
        np.testing.assert_array_almost_equal(output_grad, grad)
        
        # Input gradients are concatenated
        assert grad_x.shape == (2, 4), f"Expected shape (2, 4), got {grad_x.shape}"
        expected_grad_x = np.array([[0.1, 0.2, 0.1, 0.2], [0.3, 0.4, 0.3, 0.4]])
        np.testing.assert_array_almost_equal(grad_x, expected_grad_x)

if __name__ == "__main__":
    test_megatron_forward()
    test_megatron_backward() 