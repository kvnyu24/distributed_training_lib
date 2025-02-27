import numpy as np
from mpi4py import MPI
from distributed_training_lib import ModelParallelConfig, ParallelTrainer

def test_data_parallel_training():
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Create configuration with data parallelism
    config = ModelParallelConfig(
        mp_size=1,  # No model parallelism
        dp_size=2,  # Use 2 processes for data parallelism
        is_megatron=False,
        in_dim=4,
        out_dim=2
    )
    
    # Create different data batches for each data parallel process
    if rank == 0:
        # First half of the batch
        data = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    else:
        # Second half of the batch
        data = np.array([[9., 10., 11., 12.], [13., 14., 15., 16.]])
    
    # Initialize trainer
    trainer = ParallelTrainer(config)
    
    # Verify data parallel setup
    if rank == 0:
        print(f"Data Parallel Index: {trainer.dp_idx}")
        print(f"Model Parallel Index: {trainer.mp_idx}")
        assert trainer.dp_idx == 0, "Expected dp_idx to be 0 for first process"
        assert trainer.mp_idx == 0, "Expected mp_idx to be 0 for no model parallelism"
    else:
        print(f"Data Parallel Index: {trainer.dp_idx}")
        print(f"Model Parallel Index: {trainer.mp_idx}")
        assert trainer.dp_idx == 1, "Expected dp_idx to be 1 for second process"
        assert trainer.mp_idx == 0, "Expected mp_idx to be 0 for no model parallelism"
    
    # Test forward pass
    output = trainer.forward(data)
    
    # Each process should maintain its own batch of data
    if rank == 0:
        print(f"Rank {rank} - Input shape: {data.shape}")
        print(f"Rank {rank} - Input:\n{data}")
        print(f"Rank {rank} - Output shape: {output.shape}")
        print(f"Rank {rank} - Output:\n{output}")
        assert data.shape == (2, 4), f"Expected input shape (2, 4), got {data.shape}"
        assert output.shape == (2, 4), f"Expected output shape (2, 4), got {output.shape}"
    
    # Test backward pass
    grad = np.ones_like(output) * (rank + 1)  # Different gradients for each process
    grad_output = trainer.backward(grad)
    
    # Verify gradient shapes
    if rank == 0:
        print(f"Rank {rank} - Gradient shape: {grad_output.shape}")
        print(f"Rank {rank} - Gradient:\n{grad_output}")
        assert grad_output.shape == (2, 4), f"Expected gradient shape (2, 4), got {grad_output.shape}"

    # Synchronize gradients across data parallel processes
    all_grads = trainer.dp_comm.allgather(grad_output)
    
    if rank == 0:
        # Verify that we received gradients from all data parallel processes
        assert len(all_grads) == config.dp_size, f"Expected {config.dp_size} gradients, got {len(all_grads)}"
        print("All gradients gathered successfully")
        print(f"Number of gradient tensors: {len(all_grads)}")
        for i, g in enumerate(all_grads):
            print(f"Gradient {i} shape: {g.shape}")
            print(f"Gradient {i}:\n{g}")

if __name__ == "__main__":
    test_data_parallel_training() 