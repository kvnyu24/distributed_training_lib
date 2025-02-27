import os
import shutil
import numpy as np
from mpi4py import MPI
import tempfile
from distributed_training_lib import ModelParallelConfig, ParallelTrainer, CheckpointManager

def test_checkpoint_save_load():
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Create a temporary directory for checkpoints
    if rank == 0:
        temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {temp_dir}")
    else:
        temp_dir = None
    
    # Broadcast the directory path to all processes
    temp_dir = comm.bcast(temp_dir, root=0)
    
    # Create checkpoint manager
    checkpoint_dir = os.path.join(temp_dir, "checkpoints")
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        comm=comm,
        rank=rank
    )
    
    # Create some sample state to save
    weights = np.random.rand(5, 5)
    bias = np.random.rand(5)
    learning_rate = 0.01
    
    state_dict = {
        "weights": weights,
        "bias": bias,
        "learning_rate": learning_rate
    }
    
    # Create model parallel info
    model_parallel_info = {
        "mp_size": 2,
        "dp_size": 1,
        "is_megatron": False
    }
    
    # Save checkpoint
    step = 100
    checkpoint_path = checkpoint_manager.save_checkpoint(
        state_dict=state_dict,
        step=step,
        model_parallel_info=model_parallel_info
    )
    
    # Make sure all processes finish saving
    comm.Barrier()
    
    # Load checkpoint
    loaded_state = checkpoint_manager.load_checkpoint(step=step)
    
    if rank == 0:
        # Verify loaded state
        print(f"Original weights shape: {weights.shape}")
        print(f"Loaded weights shape: {loaded_state['weights'].shape}")
        
        assert "weights" in loaded_state, "Weights not found in loaded state"
        assert "bias" in loaded_state, "Bias not found in loaded state"
        assert "learning_rate" in loaded_state, "Learning rate not found in loaded state"
        assert "step" in loaded_state, "Step not found in loaded state"
        
        assert np.allclose(loaded_state["weights"], weights), "Loaded weights don't match original"
        assert np.allclose(loaded_state["bias"], bias), "Loaded bias doesn't match original"
        assert loaded_state["learning_rate"] == learning_rate, "Loaded learning rate doesn't match original"
        assert loaded_state["step"] == step, "Loaded step doesn't match original"
        
        # Verify model parallel info
        assert "model_parallel_info" in loaded_state, "Model parallel info not found in loaded state"
        loaded_mp_info = loaded_state["model_parallel_info"]
        assert loaded_mp_info["mp_size"] == 2, "Loaded mp_size doesn't match original"
        assert loaded_mp_info["dp_size"] == 1, "Loaded dp_size doesn't match original"
        assert loaded_mp_info["is_megatron"] == False, "Loaded is_megatron doesn't match original"
        
        print("All checkpoint tests passed!")
    
    # Save another checkpoint
    step = 200
    new_weights = np.random.rand(5, 5)
    state_dict["weights"] = new_weights
    checkpoint_manager.save_checkpoint(
        state_dict=state_dict,
        step=step
    )
    
    # Load latest checkpoint (without specifying step)
    latest_state = checkpoint_manager.load_checkpoint()
    
    if rank == 0:
        # Verify latest state is the second one we saved
        assert latest_state["step"] == 200, "Latest checkpoint should be step 200"
        assert np.allclose(latest_state["weights"], new_weights), "Latest weights don't match"
        print("Latest checkpoint test passed!")
    
    # Clean up
    if rank == 0:
        shutil.rmtree(temp_dir)
        print(f"Removed temporary directory: {temp_dir}")

if __name__ == "__main__":
    test_checkpoint_save_load() 