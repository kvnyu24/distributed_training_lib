import numpy as np
from mpi4py import MPI
import time
import os
import sys

# Add the parent directory to the path so we can import our library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from distributed_training_lib.parallel.sequence_parallel import SequenceParallelism

def test_sequence_splitting():
    """Test basic sequence splitting functionality."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping sequence splitting test: need at least 2 processes")
        return
    
    # Create sequence parallelism instance
    sp = SequenceParallelism(comm, seq_dim=1)  # Sequence dimension is 1 (batch_size, seq_len, ...)
    
    # Create a test tensor with sequence length 16
    batch_size = 2
    seq_len = 16
    hidden_dim = 4
    tensor = np.ones((batch_size, seq_len, hidden_dim)) * (rank + 1)  # Different values on each rank
    
    # Split the tensor along the sequence dimension
    local_tensor = sp.split_sequence(tensor)
    
    # Calculate expected sequence length for this rank
    expected_seq_len = seq_len // size
    if rank < seq_len % size:
        expected_seq_len += 1
    
    expected_shape = (batch_size, expected_seq_len, hidden_dim)
    
    # Verify local tensor shape
    if rank == 0:
        print(f"Rank {rank}: Local tensor shape: {local_tensor.shape}, Expected: {expected_shape}")
    assert local_tensor.shape == expected_shape, f"Incorrect local tensor shape: {local_tensor.shape}"
    
    # Gather the tensor
    gathered_tensor = sp.gather_sequence(local_tensor)
    
    # Verify gathered tensor
    if rank == 0:
        print(f"Rank {rank}: Gathered tensor shape: {gathered_tensor.shape}")
        print(f"Rank {rank}: Expected shape: {tensor.shape}")
    assert gathered_tensor.shape == tensor.shape, f"Incorrect gathered tensor shape: {gathered_tensor.shape}"
    
    if rank == 0:
        print("Sequence splitting test passed!")

def test_sequence_parallel_attention():
    """Test sequence-parallel attention computation."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping sequence-parallel attention test: need at least 2 processes")
        return
    
    # Create sequence parallelism instance
    sp = SequenceParallelism(comm)
    
    # Create test data
    np.random.seed(42)
    batch_size = 2
    seq_len = 8
    hidden_dim = 4
    
    # Create query, key, value tensors
    query = np.random.randn(batch_size, seq_len, hidden_dim)
    key = np.random.randn(batch_size, seq_len, hidden_dim)
    value = np.random.randn(batch_size, seq_len, hidden_dim)
    
    # Create attention mask (optional)
    # In this example, mask the upper triangular part (causal mask)
    mask = np.tril(np.ones((batch_size, seq_len, seq_len))) * -1e9
    mask = np.where(mask == 0, mask, -1e9)  # Convert zeros to -1e9
    
    if rank == 0:
        print(f"Rank {rank}: Running sequence-parallel attention")
    
    # Compute sequence-parallel attention
    start_time = time.time()
    attention_output = sp.sequence_parallel_attention(query, key, value, mask)
    end_time = time.time()
    
    # Verify output shape
    expected_shape = (batch_size, seq_len, hidden_dim)
    assert attention_output.shape == expected_shape, \
        f"Incorrect output shape: {attention_output.shape} vs expected {expected_shape}"
    
    if rank == 0:
        print(f"Rank {rank}: Attention computation time: {end_time - start_time:.4f}s")
        print(f"Rank {rank}: Output shape: {attention_output.shape}")
        print("Sequence-parallel attention test passed!")

def test_sequence_parallel_mlp():
    """Test sequence-parallel MLP computation."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping sequence-parallel MLP test: need at least 2 processes")
        return
    
    # Create sequence parallelism instance
    sp = SequenceParallelism(comm)
    
    # Create test data
    np.random.seed(42)
    batch_size = 2
    seq_len = 8
    hidden_dim = 4
    intermediate_dim = 8
    
    # Create input tensor and weights
    input_tensor = np.random.randn(batch_size, seq_len, hidden_dim)
    weights1 = np.random.randn(hidden_dim, intermediate_dim)
    biases1 = np.random.randn(intermediate_dim)
    weights2 = np.random.randn(intermediate_dim, hidden_dim)
    biases2 = np.random.randn(hidden_dim)
    
    if rank == 0:
        print(f"Rank {rank}: Running sequence-parallel MLP")
    
    # Compute sequence-parallel MLP
    start_time = time.time()
    mlp_output = sp.sequence_parallel_mlp(input_tensor, weights1, biases1, weights2, biases2)
    end_time = time.time()
    
    # Verify output shape
    expected_shape = (batch_size, seq_len, hidden_dim)
    assert mlp_output.shape == expected_shape, \
        f"Incorrect output shape: {mlp_output.shape} vs expected {expected_shape}"
    
    if rank == 0:
        print(f"Rank {rank}: MLP computation time: {end_time - start_time:.4f}s")
        print(f"Rank {rank}: Output shape: {mlp_output.shape}")
        print("Sequence-parallel MLP test passed!")

def test_sequence_parallel_transformer_layer():
    """Test sequence-parallel transformer layer computation."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping sequence-parallel transformer layer test: need at least 2 processes")
        return
    
    # Create sequence parallelism instance
    sp = SequenceParallelism(comm)
    
    # Create test data
    np.random.seed(42)
    batch_size = 2
    seq_len = 8
    hidden_dim = 16
    intermediate_dim = 32
    
    # Create input tensor
    input_tensor = np.random.randn(batch_size, seq_len, hidden_dim)
    
    # Create transformer layer parameters
    attention_params = {
        'wq': np.random.randn(hidden_dim, hidden_dim) * 0.02,
        'wk': np.random.randn(hidden_dim, hidden_dim) * 0.02,
        'wv': np.random.randn(hidden_dim, hidden_dim) * 0.02,
        'wo': np.random.randn(hidden_dim, hidden_dim) * 0.02,
        'bq': np.zeros(hidden_dim),
        'bk': np.zeros(hidden_dim),
        'bv': np.zeros(hidden_dim),
        'bo': np.zeros(hidden_dim)
    }
    
    mlp_params = {
        'w1': np.random.randn(hidden_dim, intermediate_dim) * 0.02,
        'w2': np.random.randn(intermediate_dim, hidden_dim) * 0.02,
        'b1': np.zeros(intermediate_dim),
        'b2': np.zeros(hidden_dim)
    }
    
    layer_norm_params = {
        'ln1_gamma': np.ones(hidden_dim),
        'ln1_beta': np.zeros(hidden_dim),
        'ln2_gamma': np.ones(hidden_dim),
        'ln2_beta': np.zeros(hidden_dim)
    }
    
    # Create attention mask (optional)
    # In this example, mask the upper triangular part (causal mask)
    mask = np.tril(np.ones((batch_size, seq_len, seq_len))) * -1e9
    mask = np.where(mask == 0, mask, -1e9)  # Convert zeros to -1e9
    
    if rank == 0:
        print(f"Rank {rank}: Running sequence-parallel transformer layer")
    
    # Compute sequence-parallel transformer layer
    start_time = time.time()
    layer_output = sp.sequence_parallel_transformer_layer(
        input_tensor, attention_params, mlp_params, layer_norm_params, mask
    )
    end_time = time.time()
    
    # Verify output shape
    expected_shape = (batch_size, seq_len, hidden_dim)
    assert layer_output.shape == expected_shape, \
        f"Incorrect output shape: {layer_output.shape} vs expected {expected_shape}"
    
    if rank == 0:
        print(f"Rank {rank}: Transformer layer computation time: {end_time - start_time:.4f}s")
        print(f"Rank {rank}: Output shape: {layer_output.shape}")
        print("Sequence-parallel transformer layer test passed!")

def test_sequence_parallel_transformer():
    """Test full sequence-parallel transformer model."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping sequence-parallel transformer test: need at least 2 processes")
        return
    
    # Create sequence parallelism instance
    sp = SequenceParallelism(comm)
    
    # Create model configuration
    config = {
        'hidden_dim': 32,
        'intermediate_dim': 64,
        'num_layers': 2,
        'vocab_size': 100
    }
    
    # Create model parameters
    model_params = sp.create_sequence_parallel_transformer(config)
    
    # Create test input
    batch_size = 2
    seq_len = 8
    input_ids = np.random.randint(0, config['vocab_size'], size=(batch_size, seq_len))
    
    # Create attention mask (causal mask)
    attention_mask = np.tril(np.ones((batch_size, seq_len, seq_len))) * -1e9
    attention_mask = np.where(attention_mask == 0, attention_mask, -1e9)
    
    if rank == 0:
        print(f"Rank {rank}: Running full sequence-parallel transformer model")
        print(f"Rank {rank}: Model has {len(model_params['layers'])} layers with hidden dim {config['hidden_dim']}")
    
    # Run sequence-parallel forward pass
    start_time = time.time()
    logits = sp.sequence_parallel_forward(input_ids, model_params, attention_mask)
    end_time = time.time()
    
    # Verify output shape
    expected_shape = (batch_size, seq_len, config['vocab_size'])
    assert logits.shape == expected_shape, \
        f"Incorrect output shape: {logits.shape} vs expected {expected_shape}"
    
    if rank == 0:
        print(f"Rank {rank}: Full transformer forward pass time: {end_time - start_time:.4f}s")
        print(f"Rank {rank}: Output logits shape: {logits.shape}")
        print("Full sequence-parallel transformer test passed!")

def test_long_sequence_scaling():
    """Test sequence parallelism scaling with long sequences."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping long sequence scaling test: need at least 2 processes")
        return
    
    # Create sequence parallelism instance
    sp = SequenceParallelism(comm)
    
    # Test with increasingly long sequences
    seq_lengths = [64, 128, 256]  # Could test with longer sequences if needed
    batch_size = 1
    hidden_dim = 32
    
    results = []
    
    for seq_len in seq_lengths:
        # Create input tensor
        input_tensor = np.random.randn(batch_size, seq_len, hidden_dim)
        
        # Split and gather to measure performance
        start_time = time.time()
        
        # Split
        local_tensor = sp.split_sequence(input_tensor)
        
        # Simulate some computation
        local_tensor = local_tensor * 2.0
        
        # Gather
        gathered_tensor = sp.gather_sequence(local_tensor)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Record results
        local_seq_len = local_tensor.shape[1]
        results.append((seq_len, local_seq_len, elapsed))
        
        if rank == 0:
            print(f"Sequence length: {seq_len}, Local sequence length: {local_seq_len}, Time: {elapsed:.4f}s")
    
    if rank == 0:
        print("\nScaling results:")
        for seq_len, local_seq_len, elapsed in results:
            print(f"Sequence length: {seq_len}, " 
                 f"Local sequence length: {local_seq_len} ({local_seq_len/seq_len:.2%}), "
                 f"Time: {elapsed:.4f}s")
        
        print("Long sequence scaling test passed!")

if __name__ == "__main__":
    print("\n==== Sequence Splitting Test ====")
    test_sequence_splitting()
    
    print("\n==== Sequence-Parallel Attention Test ====")
    test_sequence_parallel_attention()
    
    print("\n==== Sequence-Parallel MLP Test ====")
    test_sequence_parallel_mlp()
    
    print("\n==== Sequence-Parallel Transformer Layer Test ====")
    test_sequence_parallel_transformer_layer()
    
    print("\n==== Full Sequence-Parallel Transformer Test ====")
    test_sequence_parallel_transformer()
    
    print("\n==== Long Sequence Scaling Test ====")
    test_long_sequence_scaling()
    
    print("\n==== All sequence parallelism tests completed! ====\n") 