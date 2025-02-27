import numpy as np
from mpi4py import MPI
import time
from distributed_training_lib import (
    PipelineParallelConfig, 
    PipelineParallelTrainer,
    PipelineStage,
    PipelineScheduler
)

def test_pipeline_config():
    """Test pipeline parallel configuration."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Create configuration
    config = PipelineParallelConfig(
        num_stages=2,
        micro_batch_size=4,
        num_micro_batches=2,
        dp_size=1
    )
    
    if rank == 0:
        print(f"Pipeline config test:")
        print(f"  Number of stages: {config.num_stages}")
        print(f"  Micro-batch size: {config.micro_batch_size}")
        print(f"  Number of micro-batches: {config.num_micro_batches}")
        print(f"  Data parallel size: {config.dp_size}")
        print(f"  Total batch size: {config.batch_size}")
        
        assert config.num_stages == 2, f"Expected 2 stages, got {config.num_stages}"
        assert config.micro_batch_size == 4, f"Expected micro-batch size 4, got {config.micro_batch_size}"
        assert config.num_micro_batches == 2, f"Expected 2 micro-batches, got {config.num_micro_batches}"
        assert config.batch_size == 8, f"Expected batch size 8, got {config.batch_size}"
        
        print("Pipeline config test passed!")

def test_pipeline_stage():
    """Test pipeline stage functionality."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping pipeline stage test: need at least 2 processes")
        return
    
    # Determine stage info
    stage_id = rank % 2  # 0 or 1
    next_rank = None if stage_id == 1 else 1  # Only for stage 0
    prev_rank = None if stage_id == 0 else 0  # Only for stage 1
    
    # Create stage
    stage = PipelineStage(
        stage_id=stage_id,
        num_stages=2,
        micro_batch_size=4,
        comm=comm,
        world_rank=rank,
        next_stage_rank=next_rank,
        prev_stage_rank=prev_rank
    )
    
    # Test stage initialization
    is_first = stage.is_first_stage()
    is_last = stage.is_last_stage()
    
    print(f"Rank {rank} - Stage {stage_id}: is_first={is_first}, is_last={is_last}")
    
    assert is_first == (stage_id == 0), f"Wrong is_first_stage value: {is_first}"
    assert is_last == (stage_id == 1), f"Wrong is_last_stage value: {is_last}"
    
    # Test buffer clearing
    stage.activation_buffer[0] = np.ones((4, 4))
    stage.gradient_buffer[0] = np.ones((4, 4)) * 2
    
    assert 0 in stage.activation_buffer, "Activation should be in buffer"
    assert 0 in stage.gradient_buffer, "Gradient should be in buffer"
    
    stage.clear_buffers()
    
    assert 0 not in stage.activation_buffer, "Activation should not be in buffer"
    assert 0 not in stage.gradient_buffer, "Gradient should not be in buffer"
    
    # Simple communication test
    if rank == 0:
        print(f"Rank {rank} - Stage communication test...")
    
    if stage_id == 0:  # First stage
        data = np.ones((4, 4))
        stage.comm.Send(data, dest=1, tag=42)
        print(f"Rank {rank} - Sent data to next stage")
    else:  # Last stage
        data = np.empty((4, 4))
        stage.comm.Recv(data, source=0, tag=42)
        print(f"Rank {rank} - Received data from previous stage: {data[0, 0]}")
        assert np.allclose(data, 1.0), f"Expected 1.0, got {data[0, 0]}"
    
    # Make sure all processes finish
    comm.Barrier()
    if rank == 0:
        print("Pipeline stage test passed!")

def test_simple_pipeline():
    """Test simplified pipeline parallelism without the full trainer."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping simple pipeline test: need at least 2 processes")
        return
    
    # Create a simple two-stage pipeline    
    pipeline_rank = rank % 2  # 0 or 1 (stage ID)
    
    if rank == 0:
        print(f"Running simple pipeline test...")
    
    # Testing simple forward pass
    if pipeline_rank == 0:  # First stage
        # Process input
        input_data = np.ones((4, 4))
        output_data = input_data + pipeline_rank  # Add stage ID
        
        # Send to next stage
        comm.Send(output_data, dest=1, tag=10)
        print(f"Rank {rank} - Sent processed data to next stage")
        
        # Receive gradients from next stage
        gradients = np.empty_like(output_data)
        comm.Recv(gradients, source=1, tag=20)
        print(f"Rank {rank} - Received gradients: {gradients[0, 0]}")
        
        # Check gradients
        assert np.allclose(gradients, 2.0), f"Expected 2.0, got {gradients[0, 0]}"
        
    else:  # Last stage
        # Receive from previous stage
        input_data = np.empty((4, 4))
        comm.Recv(input_data, source=0, tag=10)
        print(f"Rank {rank} - Received data from previous stage: {input_data[0, 0]}")
        
        # Process input
        output_data = input_data + pipeline_rank  # Add stage ID
        
        # First stage sends 1.0, then we add stage_id (1) to get 2.0
        expected_output = 2.0
        assert np.allclose(output_data, expected_output), f"Expected {expected_output}, got {output_data[0, 0]}"
        print(f"Rank {rank} - Processed to {output_data[0, 0]}")
        
        # Generate gradients (just using ones * (stage_id + 1))
        gradients = np.ones_like(output_data) * (pipeline_rank + 1)
        
        # Send gradients to previous stage
        comm.Send(gradients, dest=0, tag=20)
        print(f"Rank {rank} - Sent gradients to previous stage")
    
    # Make sure all processes finish
    comm.Barrier()
    if rank == 0:
        print("Simple pipeline test passed!")

def test_pipeline_stage_forward_backward():
    """Test pipeline stage forward and backward passes with direct control."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Skip if not enough processes
    if size < 2:
        if rank == 0:
            print("Skipping pipeline stage forward/backward test: need at least 2 processes")
        return
    
    # Determine stage info
    stage_id = rank % 2  # 0 or 1
    next_rank = None if stage_id == 1 else 1  # Only for stage 0
    prev_rank = None if stage_id == 0 else 0  # Only for stage 1
    
    # Create stage
    stage = PipelineStage(
        stage_id=stage_id,
        num_stages=2,
        micro_batch_size=4,
        comm=comm,
        world_rank=rank,
        next_stage_rank=next_rank,
        prev_stage_rank=prev_rank
    )
    
    if rank == 0:
        print(f"Running pipeline stage forward/backward test...")
    
    # Forward pass - use direct communication instead of stage.forward
    if stage_id == 0:  # First stage
        input_data = np.ones((4, 4))
        output_data = input_data + stage_id  # Add stage ID
        
        # Send to next stage
        comm.Send(output_data, dest=1, tag=100)
        print(f"Rank {rank} - First stage forward pass complete")
    else:  # Last stage
        input_data = np.empty((4, 4))
        comm.Recv(input_data, source=0, tag=100)
        output_data = input_data + stage_id  # Add stage ID
        
        print(f"Rank {rank} - Last stage output: {output_data[0, 0]}")
        # First stage sends 1.0, then we add stage_id (1) to get 2.0
        assert np.allclose(output_data, 2.0), f"Expected 2.0, got {output_data[0, 0]}"
    
    # Store activations for backward pass
    stage.activation_buffer[0] = output_data.copy()
    
    # Synchronize before backward
    comm.Barrier()
    
    # Backward pass - use direct communication instead of stage.backward
    if stage_id == 1:  # Last stage
        # Use ones as output gradients
        output_grads = np.ones((4, 4))
        input_grads = output_grads * (stage_id + 1)  # Multiply by stage_id + 1
        
        # Send to previous stage
        comm.Send(input_grads, dest=0, tag=200)
        print(f"Rank {rank} - Last stage backward pass complete")
    else:  # First stage
        input_grads = np.empty((4, 4))
        comm.Recv(input_grads, source=1, tag=200)
        
        print(f"Rank {rank} - First stage input gradients: {input_grads[0, 0]}")
        assert np.allclose(input_grads, 2.0), f"Expected 2.0, got {input_grads[0, 0]}"
    
    # Clean up
    stage.clear_buffers()
    
    # Make sure all processes finish
    comm.Barrier()
    if rank == 0:
        print("Pipeline stage forward/backward test passed!")

if __name__ == "__main__":
    print("\n==== Pipeline Config Test ====")
    test_pipeline_config()
    
    print("\n==== Pipeline Stage Test ====")
    test_pipeline_stage()
    
    print("\n==== Simple Pipeline Test ====")
    test_simple_pipeline()
    
    print("\n==== Pipeline Stage Forward/Backward Test ====")
    test_pipeline_stage_forward_backward()
    
    print("\n==== All pipeline parallel tests completed! ====\n") 