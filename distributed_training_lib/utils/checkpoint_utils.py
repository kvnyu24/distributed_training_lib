import os
import json
import numpy as np
import h5py
from mpi4py import MPI
from typing import Dict, Any, Union, Optional

class CheckpointManager:
    """Class to manage checkpoint saving and loading in distributed training."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        comm: MPI.Comm,
        rank: int,
        max_to_keep: int = 5
    ):
        """Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            comm: MPI communicator
            rank: Process rank
            max_to_keep: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.comm = comm
        self.rank = rank
        self.max_to_keep = max_to_keep
        self.checkpoint_files = []
        
        # Create checkpoint directory if it doesn't exist
        if self.rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(
        self, 
        state_dict: Dict[str, Any], 
        step: int, 
        model_parallel_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a checkpoint with model state and optimizer state.
        
        Args:
            state_dict: Dictionary containing model weights and other state
            step: Current training step
            model_parallel_info: Information about model parallelism configuration
            
        Returns:
            Path to the saved checkpoint
        """
        # Ensure only rank 0 writes the checkpoint metadata
        if self.rank == 0:
            # Create checkpoint filename
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}.h5")
            
            # Save checkpoint
            with h5py.File(checkpoint_path, 'w') as f:
                # Save model parameters and other state
                for key, value in state_dict.items():
                    if isinstance(value, np.ndarray):
                        f.create_dataset(key, data=value)
                    elif isinstance(value, (int, float, str)):
                        f.attrs[key] = value
                
                # Save model parallel info
                if model_parallel_info:
                    info_group = f.create_group('model_parallel_info')
                    for key, value in model_parallel_info.items():
                        info_group.attrs[key] = value
                
                # Save step information
                f.attrs['step'] = step
            
            # Update checkpoint list
            self.checkpoint_files.append(checkpoint_path)
            if len(self.checkpoint_files) > self.max_to_keep:
                old_checkpoint = self.checkpoint_files.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
            
            # Save checkpoint metadata
            self._save_checkpoint_metadata()
            
            print(f"Checkpoint saved at {checkpoint_path}")
            return checkpoint_path
        
        # Other ranks just wait for rank 0 to finish
        self.comm.Barrier()
        return ""
    
    def load_checkpoint(self, step: Optional[int] = None) -> Dict[str, Any]:
        """Load a checkpoint.
        
        Args:
            step: Specific step to load, or None for latest
            
        Returns:
            Dictionary with loaded state
        """
        # Get checkpoint metadata
        checkpoint_metadata = self._load_checkpoint_metadata()
        
        if not checkpoint_metadata:
            print("No checkpoints found")
            return {}
        
        # Determine which checkpoint to load
        if step is None:
            checkpoint_path = checkpoint_metadata['latest_checkpoint']
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}.h5")
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint for step {step} not found")
                return {}
        
        # Load checkpoint
        state_dict = {}
        with h5py.File(checkpoint_path, 'r') as f:
            # Load step information
            state_dict['step'] = f.attrs['step']
            
            # Load model parallel info if exists
            if 'model_parallel_info' in f:
                model_parallel_info = {}
                for key, value in f['model_parallel_info'].attrs.items():
                    model_parallel_info[key] = value
                state_dict['model_parallel_info'] = model_parallel_info
            
            # Load datasets
            for key in f.keys():
                if key != 'model_parallel_info':
                    state_dict[key] = f[key][()]
            
            # Load attributes
            for key, value in f.attrs.items():
                if key != 'step':
                    state_dict[key] = value
        
        if self.rank == 0:
            print(f"Checkpoint loaded from {checkpoint_path}")
        
        return state_dict
    
    def _save_checkpoint_metadata(self):
        """Save metadata about available checkpoints."""
        metadata_path = os.path.join(self.checkpoint_dir, "checkpoint_metadata.json")
        metadata = {
            'all_checkpoints': self.checkpoint_files,
            'latest_checkpoint': self.checkpoint_files[-1] if self.checkpoint_files else None
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    def _load_checkpoint_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata.
        
        Returns:
            Dictionary with checkpoint metadata
        """
        metadata_path = os.path.join(self.checkpoint_dir, "checkpoint_metadata.json")
        if not os.path.exists(metadata_path):
            return {}
        
        with open(metadata_path, 'r') as f:
            return json.load(f) 