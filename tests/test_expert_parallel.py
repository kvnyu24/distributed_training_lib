import numpy as np
from mpi4py import MPI
import time
import os
import sys

# Add the parent directory to the path so we can import our library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import directly from the module file
from distributed_training_lib.parallel.expert_parallel import ExpertParallelism

def test_expert_distribution():
    """Test the distribution of experts across ranks."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping expert distribution test: need at least 2 processes")
        return
    
    # Create expert parallelism instance with 8 experts
    num_experts = 8
    ep = ExpertParallelism(comm, num_experts)
    
    # Verify expert distribution
    expected_experts_per_rank = num_experts // size
    remainder = num_experts % size
    
    expected_num_experts = expected_experts_per_rank
    if rank < remainder:
        expected_num_experts += 1
    
    if rank == 0:
        print(f"Rank {rank}: Expected {expected_num_experts} experts, got {len(ep.local_expert_indices)}")
        
    assert len(ep.local_expert_indices) == expected_num_experts, \
        f"Incorrect number of experts assigned to rank {rank}"
    
    # Verify that each expert is assigned to exactly one rank
    all_experts = []
    for r in range(size):
        if r == rank:
            all_experts.append(ep.local_expert_indices)
        else:
            all_experts.append([])
    
    all_experts = comm.allgather(ep.local_expert_indices)
    
    # Flatten list of lists
    all_experts_flat = [e for sublist in all_experts for e in sublist]
    
    # Check that all experts are assigned
    all_experts_flat.sort()
    assert all_experts_flat == list(range(num_experts)), \
        f"Not all experts were assigned or some were assigned multiple times"
    
    if rank == 0:
        print("Expert distribution test passed!")

def test_dispatch_combine():
    """Test the dispatch and combine operations."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping dispatch/combine test: need at least 2 processes")
        return
    
    # Create expert parallelism instance with 4 experts
    num_experts = 4
    ep = ExpertParallelism(comm, num_experts)
    
    # Create a batch of tokens
    batch_size = 8
    hidden_dim = 4
    
    # Create token embeddings (different on each rank to verify correct routing)
    np.random.seed(42 + rank)
    input_tensor = np.random.randn(batch_size, hidden_dim)
    
    # Create expert assignments for each token (top-2 experts)
    k = 2
    expert_indices = np.zeros((batch_size, k), dtype=np.int32)
    expert_weights = np.zeros((batch_size, k), dtype=np.float32)
    
    # Assign each token to 2 experts with equal weights
    for i in range(batch_size):
        # Deterministic assignment for testing
        expert_indices[i, 0] = i % num_experts
        expert_indices[i, 1] = (i + 1) % num_experts
        expert_weights[i, 0] = 0.6
        expert_weights[i, 1] = 0.4
    
    if rank == 0:
        print(f"Rank {rank}: Testing dispatch with batch_size={batch_size}, hidden_dim={hidden_dim}")
        print(f"Rank {rank}: Expert indices shape: {expert_indices.shape}")
        
    # Dispatch tokens to experts
    local_input, local_expert_indices, local_token_indices, local_weights = ep.dispatch(
        input_tensor, expert_indices, expert_weights
    )
    
    if rank == 0:
        print(f"Rank {rank}: Local input shape: {local_input.shape}")
        print(f"Rank {rank}: Local expert indices: {local_expert_indices}")
        print(f"Rank {rank}: Local token indices: {local_token_indices}")
    
    # Simulate processing by experts (just identity function for testing)
    local_output = local_input.copy()
    
    # Combine results back
    combined_output = ep.combine(
        local_output, local_token_indices, local_weights, batch_size, hidden_dim
    )
    
    # Verify combined output shape
    assert combined_output.shape == (batch_size, hidden_dim), \
        f"Incorrect combined output shape: got {combined_output.shape}, expected {(batch_size, hidden_dim)}"
    
    if rank == 0:
        print(f"Rank {rank}: Combined output shape: {combined_output.shape}")
        print("Dispatch and combine test passed!")

def test_moe_layer():
    """Test the MoE layer creation and forward pass."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping MoE layer test: need at least 2 processes")
        return
    
    # Create expert parallelism instance with 4 experts
    num_experts = 4
    ep = ExpertParallelism(comm, num_experts)
    
    # Create MoE layer
    hidden_size = 8
    ffn_hidden_size = 16
    moe_layer = ep.create_moe_layer(hidden_size, ffn_hidden_size)
    
    if rank == 0:
        print(f"Rank {rank}: Created MoE layer with {len(moe_layer)} local experts")
        for expert_idx, params in moe_layer.items():
            print(f"  Expert {expert_idx}: w1 shape {params['w1'].shape}, w2 shape {params['w2'].shape}")
    
    # Create a batch of tokens
    batch_size = 6
    
    # Create token embeddings
    np.random.seed(42)
    input_tensor = np.random.randn(batch_size, hidden_size)
    
    # Create expert assignments for each token (top-2 experts)
    k = 2
    expert_indices = np.zeros((batch_size, k), dtype=np.int32)
    expert_weights = np.zeros((batch_size, k), dtype=np.float32)
    
    # Assign each token to 2 experts with equal weights
    for i in range(batch_size):
        # Deterministic assignment for testing
        expert_indices[i, 0] = i % num_experts
        expert_indices[i, 1] = (i + 1) % num_experts
        expert_weights[i, 0] = 0.7
        expert_weights[i, 1] = 0.3
    
    # Forward pass through MoE layer
    output_tensor = ep.forward(input_tensor, expert_indices, expert_weights, moe_layer)
    
    # Verify output shape
    assert output_tensor.shape == (batch_size, hidden_size), \
        f"Incorrect output shape: got {output_tensor.shape}, expected {(batch_size, hidden_size)}"
    
    if rank == 0:
        print(f"Rank {rank}: MoE layer forward output shape: {output_tensor.shape}")
        print("MoE layer test passed!")

if __name__ == "__main__":
    print("\n==== Expert Distribution Test ====")
    test_expert_distribution()
    
    print("\n==== Dispatch and Combine Test ====")
    test_dispatch_combine()
    
    print("\n==== MoE Layer Test ====")
    test_moe_layer()
    
    print("\n==== All expert parallelism tests completed! ====\n")