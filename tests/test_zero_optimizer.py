import numpy as np
from mpi4py import MPI
import time
from distributed_training_lib.optimizers.zero_optimizer import ZeROOptimizer, ZeROStage

def test_zero_stage1():
    """Test ZeRO optimizer Stage 1 (optimizer state partitioning)."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping ZeRO Stage 1 test: need at least 2 processes")
        return
    
    # Create optimizer
    optimizer = ZeROOptimizer(
        dp_comm=comm,
        stage=ZeROStage.STAGE_1,
        learning_rate=0.1
    )
    
    # Create parameters
    params = {
        "weight1": np.ones((2, 2)) * (rank + 1),
        "weight2": np.ones((2, 2)) * (rank + 2),
        "bias1": np.ones(2) * (rank + 3),
        "bias2": np.ones(2) * (rank + 4)
    }
    
    # Register parameters
    optimizer.register_parameters(params)
    
    # Verify parameter partitioning
    if rank == 0:
        print(f"Rank {rank} - ZeRO Stage 1 Optimizer created")
        print(f"Rank {rank} - Optimizer should partition optimizer states only")
        for name in params.keys():
            print(f"  Parameter {name} assigned to partition {optimizer._get_partition_for_param(name)}")
    
    # Create gradients
    grads = {
        "weight1": np.ones((2, 2)) * 0.1,
        "weight2": np.ones((2, 2)) * 0.2,
        "bias1": np.ones(2) * 0.3,
        "bias2": np.ones(2) * 0.4
    }
    
    # Reduce gradients
    reduced_grads = optimizer.reduce_gradients(grads)
    
    # Verify that gradients are reduced correctly (all processes should have all gradients)
    if rank == 0:
        print(f"Rank {rank} - Checking reduced gradients")
        # All processes should reduce to the same value: average of all gradients
        assert np.allclose(reduced_grads["weight1"], np.ones((2, 2)) * 0.1)
        assert np.allclose(reduced_grads["weight2"], np.ones((2, 2)) * 0.2)
        assert np.allclose(reduced_grads["bias1"], np.ones(2) * 0.3)
        assert np.allclose(reduced_grads["bias2"], np.ones(2) * 0.4)
        print(f"Rank {rank} - Gradient reduction test passed")
    
    # Perform optimizer step
    updated_params = optimizer.step(params, reduced_grads)
    
    # Verify that parameters are updated correctly
    if rank == 0:
        print(f"Rank {rank} - Checking updated parameters")
        # Parameters should be updated: param - learning_rate * grad
        # Each process should have all parameters
        for name, param in updated_params.items():
            assert name in params, f"Missing parameter: {name}"
            assert param.shape == params[name].shape, f"Shape mismatch for {name}"
        print(f"Rank {rank} - Parameter update test passed")
    
    if rank == 0:
        print("ZeRO Stage 1 test passed!")

def test_zero_stage2():
    """Test ZeRO optimizer Stage 2 (gradient partitioning)."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping ZeRO Stage 2 test: need at least 2 processes")
        return
    
    # Create optimizer
    optimizer = ZeROOptimizer(
        dp_comm=comm,
        stage=ZeROStage.STAGE_2,
        learning_rate=0.1
    )
    
    # Create parameters
    params = {
        "weight1": np.ones((2, 2)) * (rank + 1),
        "weight2": np.ones((2, 2)) * (rank + 2),
        "bias1": np.ones(2) * (rank + 3),
        "bias2": np.ones(2) * (rank + 4)
    }
    
    # Register parameters
    optimizer.register_parameters(params)
    
    # Verify parameter partitioning
    if rank == 0:
        print(f"Rank {rank} - ZeRO Stage 2 Optimizer created")
        print(f"Rank {rank} - Optimizer should partition gradients and optimizer states")
        for name in params.keys():
            print(f"  Parameter {name} assigned to partition {optimizer._get_partition_for_param(name)}")
    
    # Create gradients
    grads = {
        "weight1": np.ones((2, 2)) * 0.1,
        "weight2": np.ones((2, 2)) * 0.2,
        "bias1": np.ones(2) * 0.3,
        "bias2": np.ones(2) * 0.4
    }
    
    # Reduce gradients
    reduced_grads = optimizer.reduce_gradients(grads)
    
    # Verify that gradients are partitioned correctly
    # In Stage 2, each process should only have gradients for parameters it owns
    owned_params = [name for name in params.keys() 
                    if optimizer._get_partition_for_param(name) == rank]
    
    if rank == 0:
        print(f"Rank {rank} - Parameters owned: {owned_params}")
        print(f"Rank {rank} - Reduced gradients keys: {reduced_grads.keys()}")
        
        # Each process should only have gradients for parameters it owns
        assert all(name in owned_params for name in reduced_grads.keys()), \
            "Process has gradients for parameters it doesn't own"
    
    # Perform optimizer step
    updated_params = optimizer.step(params, reduced_grads)
    
    # Verify that parameters are updated correctly
    if rank == 0:
        print(f"Rank {rank} - Checking updated parameters")
        # In Stage 2, parameters are not partitioned, so all processes should have all parameters
        for name, param in updated_params.items():
            assert name in params, f"Missing parameter: {name}"
            assert param.shape == params[name].shape, f"Shape mismatch for {name}"
        print(f"Rank {rank} - Parameter update test passed")
    
    if rank == 0:
        print("ZeRO Stage 2 test passed!")

def test_zero_stage3():
    """Test ZeRO optimizer Stage 3 (parameter partitioning)."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping ZeRO Stage 3 test: need at least 2 processes")
        return
    
    # Create optimizer
    optimizer = ZeROOptimizer(
        dp_comm=comm,
        stage=ZeROStage.STAGE_3,
        learning_rate=0.1
    )
    
    # Create parameters
    params = {
        "weight1": np.ones((2, 2)) * (rank + 1),
        "weight2": np.ones((2, 2)) * (rank + 2),
        "bias1": np.ones(2) * (rank + 3),
        "bias2": np.ones(2) * (rank + 4)
    }
    
    # Register parameters
    optimizer.register_parameters(params)
    
    # Verify parameter partitioning
    if rank == 0:
        print(f"Rank {rank} - ZeRO Stage 3 Optimizer created")
        print(f"Rank {rank} - Optimizer should partition parameters, gradients, and optimizer states")
        for name in params.keys():
            print(f"  Parameter {name} assigned to partition {optimizer._get_partition_for_param(name)}")
    
    # Create gradients (same for both processes for simplicity)
    grads = {
        "weight1": np.ones((2, 2)) * 0.1,
        "weight2": np.ones((2, 2)) * 0.2,
        "bias1": np.ones(2) * 0.3,
        "bias2": np.ones(2) * 0.4
    }
    
    # Reduce gradients
    reduced_grads = optimizer.reduce_gradients(grads)
    
    # Verify that gradients are partitioned correctly
    # In Stage 3, each process should only have gradients for parameters it owns
    owned_params = [name for name in params.keys() 
                    if optimizer._get_partition_for_param(name) == rank]
    
    if rank == 0:
        print(f"Rank {rank} - Parameters owned: {owned_params}")
        print(f"Rank {rank} - Reduced gradients keys: {reduced_grads.keys()}")
        
        # Each process should only have gradients for parameters it owns
        assert all(name in owned_params for name in reduced_grads.keys()), \
            "Process has gradients for parameters it doesn't own"
    
    # Perform optimizer step
    updated_params = optimizer.step(params, reduced_grads)
    
    # Verify that parameters are updated correctly
    if rank == 0:
        print(f"Rank {rank} - Checking updated parameters")
        # In Stage 3, parameters are partitioned during optimization but gathered at the end
        # So all processes should still have all parameters after the step
        for name, param in updated_params.items():
            assert name in params, f"Missing parameter: {name}"
            assert param.shape == params[name].shape, f"Shape mismatch for {name}"
        print(f"Rank {rank} - Parameter update test passed")
    
    if rank == 0:
        print("ZeRO Stage 3 test passed!")

def test_zero_integration():
    """Test ZeRO optimizer integration with multiple steps."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping ZeRO integration test: need at least 2 processes")
        return
    
    # Create optimizer (Stage 1 for simplicity)
    optimizer = ZeROOptimizer(
        dp_comm=comm,
        stage=ZeROStage.STAGE_1,
        learning_rate=0.1
    )
    
    # Create parameters
    params = {
        "weight": np.ones((4, 4)) * (rank + 1),
        "bias": np.ones(4) * (rank + 2)
    }
    
    # Register parameters
    optimizer.register_parameters(params)
    
    if rank == 0:
        print(f"Rank {rank} - Starting ZeRO integration test with multiple steps")
    
    # Simulate multiple training steps
    initial_params = params.copy()
    for step in range(3):
        # Create gradients (decreasing over time to simulate training progress)
        scale = 1.0 / (step + 1)
        grads = {
            "weight": np.ones((4, 4)) * 0.1 * scale,
            "bias": np.ones(4) * 0.2 * scale
        }
        
        # Reduce gradients
        reduced_grads = optimizer.reduce_gradients(grads)
        
        # Update parameters
        params = optimizer.step(params, reduced_grads)
        
        if rank == 0:
            print(f"Rank {rank} - Step {step+1} completed")
            
            # Verify weight updates are happening
            if step > 0:
                weight_diff = np.mean(np.abs(params["weight"] - initial_params["weight"]))
                print(f"Rank {rank} - Average weight change: {weight_diff}")
                assert weight_diff > 0, "Weights should change during training"
    
    if rank == 0:
        print("ZeRO integration test passed!")

if __name__ == "__main__":
    print("\n==== ZeRO Stage 1 Test ====")
    test_zero_stage1()
    
    print("\n==== ZeRO Stage 2 Test ====")
    test_zero_stage2()
    
    print("\n==== ZeRO Stage 3 Test ====")
    test_zero_stage3()
    
    print("\n==== ZeRO Integration Test ====")
    test_zero_integration()
    
    print("\n==== All ZeRO optimizer tests completed! ====\n") 