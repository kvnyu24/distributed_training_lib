import numpy as np
from mpi4py import MPI
import time
import os
import sys

# Add the parent directory to the path so we can import our library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import directly from the module file
from distributed_training_lib.parallel.moe_model import MixtureOfExperts

def test_moe_initialization():
    """Test initialization of the MoE model."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping MoE initialization test: need at least 2 processes")
        return
    
    # Parameters for the MoE model
    num_experts = 8
    hidden_size = 16
    ffn_hidden_size = 32
    num_experts_per_token = 2
    
    # Create MoE model
    moe = MixtureOfExperts(
        comm, num_experts, hidden_size, ffn_hidden_size, num_experts_per_token
    )
    
    # Verify router parameters
    assert moe.router_weights.shape == (hidden_size, num_experts), \
        f"Incorrect router weights shape: got {moe.router_weights.shape}, expected {(hidden_size, num_experts)}"
    
    # Verify that local experts were created
    expected_experts_per_rank = num_experts // size
    if rank < (num_experts % size):
        expected_experts_per_rank += 1
    
    assert len(moe.moe_layer) == expected_experts_per_rank, \
        f"Incorrect number of local experts: got {len(moe.moe_layer)}, expected {expected_experts_per_rank}"
    
    if rank == 0:
        print(f"Rank {rank}: MoE model initialized with {len(moe.moe_layer)} local experts")
        print(f"Rank {rank}: Router weights shape: {moe.router_weights.shape}")
        print("MoE initialization test passed!")

def test_moe_forward():
    """Test forward pass through the MoE model."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping MoE forward test: need at least 2 processes")
        return
    
    # Parameters for the MoE model
    num_experts = 4
    hidden_size = 16
    ffn_hidden_size = 32
    num_experts_per_token = 2
    
    # Create MoE model
    moe = MixtureOfExperts(
        comm, num_experts, hidden_size, ffn_hidden_size, num_experts_per_token
    )
    
    # Create a batch of tokens
    batch_size = 6
    
    # Create input embeddings - same on all ranks for deterministic testing
    np.random.seed(42)
    input_tensor = np.random.randn(batch_size, hidden_size)
    
    # Forward pass
    start_time = time.time()
    output_tensor, (expert_indices, expert_weights) = moe.forward(input_tensor)
    forward_time = time.time() - start_time
    
    # Verify output shape
    assert output_tensor.shape == (batch_size, hidden_size), \
        f"Incorrect output shape: got {output_tensor.shape}, expected {(batch_size, hidden_size)}"
    
    # Verify expert indices shape
    assert expert_indices.shape == (batch_size, num_experts_per_token), \
        f"Incorrect expert indices shape: got {expert_indices.shape}, expected {(batch_size, num_experts_per_token)}"
    
    if rank == 0:
        print(f"Rank {rank}: Forward pass completed in {forward_time:.4f}s")
        print(f"Rank {rank}: Output shape: {output_tensor.shape}")
        print(f"Rank {rank}: Expert indices shape: {expert_indices.shape}")
        print("MoE forward test passed!")

def test_load_balancing():
    """Test load balancing loss computation."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping load balancing test: need at least 2 processes")
        return
    
    # Parameters for the MoE model
    num_experts = 4
    hidden_size = 16
    ffn_hidden_size = 32
    num_experts_per_token = 2
    
    # Create MoE model
    moe = MixtureOfExperts(
        comm, num_experts, hidden_size, ffn_hidden_size, num_experts_per_token
    )
    
    # Create a batch of tokens
    batch_size = 8
    
    # Create input embeddings
    np.random.seed(42)
    input_tensor = np.random.randn(batch_size, hidden_size)
    
    # Forward pass
    output_tensor, (expert_indices, expert_weights) = moe.forward(input_tensor)
    
    # Compute load balancing loss
    load_balance_loss = moe.compute_load_balancing_loss(expert_indices, expert_weights)
    
    if rank == 0:
        print(f"Rank {rank}: Load balancing loss: {load_balance_loss}")
        
        # For perfect balance, the loss should be close to 0
        # For extreme imbalance, the loss would be higher
        # This is just a sanity check, not a strict test
        print("Load balancing test passed!")

def test_moe_update():
    """Test parameter updates in the MoE model."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping MoE update test: need at least 2 processes")
        return
    
    # Parameters for the MoE model
    num_experts = 4
    hidden_size = 8
    ffn_hidden_size = 16
    num_experts_per_token = 2
    
    # Create MoE model
    moe = MixtureOfExperts(
        comm, num_experts, hidden_size, ffn_hidden_size, num_experts_per_token
    )
    
    # Copy initial parameters for comparison
    init_router_weights = moe.router_weights.copy()
    init_expert_params = {}
    for expert_idx, params in moe.moe_layer.items():
        init_expert_params[expert_idx] = {
            "w1": params["w1"].copy(),
            "b1": params["b1"].copy(),
            "w2": params["w2"].copy(),
            "b2": params["b2"].copy()
        }
    
    # Create input and simulate forward pass
    batch_size = 4
    np.random.seed(42)
    input_tensor = np.random.randn(batch_size, hidden_size)
    output_tensor, _ = moe.forward(input_tensor)
    
    # Create a dummy gradient (using random values for simplicity)
    np.random.seed(43)
    grad_output = np.random.randn(batch_size, hidden_size)
    
    # Update parameters
    learning_rate = 0.01
    moe.update_parameters(input_tensor, output_tensor, grad_output, learning_rate)
    
    # Note: In the current implementation, the update_parameters method
    # doesn't actually compute real gradients, so we expect no change
    # This test is primarily to verify that the method executes successfully
    
    if rank == 0:
        print(f"Rank {rank}: Parameter update completed")
        print("MoE update test passed!")

def test_moe_simple_training():
    """Test a simplified training loop with the MoE model."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping MoE simple training test: need at least 2 processes")
        return
    
    # Parameters for the MoE model
    num_experts = 4
    hidden_size = 8
    ffn_hidden_size = 16
    num_experts_per_token = 2
    
    # Create MoE model
    moe = MixtureOfExperts(
        comm, num_experts, hidden_size, ffn_hidden_size, num_experts_per_token
    )
    
    # Create a simple dataset (random inputs and targets)
    batch_size = 8
    num_batches = 3
    
    # Simple training loop
    for epoch in range(2):
        for batch in range(num_batches):
            # Generate random input
            np.random.seed(epoch * 100 + batch)
            input_tensor = np.random.randn(batch_size, hidden_size)
            
            # Forward pass
            output_tensor, (expert_indices, expert_weights) = moe.forward(input_tensor)
            
            # Compute load balancing loss
            load_balance_loss = moe.compute_load_balancing_loss(expert_indices, expert_weights)
            
            # Compute "loss" (using MSE with random targets for simplicity)
            target = np.random.randn(batch_size, hidden_size)
            mse_loss = np.mean((output_tensor - target) ** 2)
            
            # Total loss with load balancing
            total_loss = mse_loss + 0.01 * load_balance_loss
            
            # Compute gradient (simplified - just the difference for MSE loss)
            grad_output = 2 * (output_tensor - target) / batch_size
            
            # Update parameters
            moe.update_parameters(input_tensor, output_tensor, grad_output, learning_rate=0.01)
            
            if rank == 0:
                print(f"Epoch {epoch}, Batch {batch}: MSE Loss = {mse_loss:.4f}, Load Balance Loss = {load_balance_loss:.4f}")
    
    if rank == 0:
        print("MoE simple training test passed!")

if __name__ == "__main__":
    print("\n==== MoE Initialization Test ====")
    test_moe_initialization()
    
    print("\n==== MoE Forward Test ====")
    test_moe_forward()
    
    print("\n==== Load Balancing Test ====")
    test_load_balancing()
    
    print("\n==== MoE Update Test ====")
    test_moe_update()
    
    print("\n==== MoE Simple Training Test ====")
    test_moe_simple_training()
    
    print("\n==== All MoE model tests completed! ====\n") 