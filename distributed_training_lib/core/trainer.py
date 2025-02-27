from typing import Optional, Tuple
import numpy as np
from mpi4py import MPI

from ..core.config import ModelParallelConfig
from ..parallel.parallel_ops import ParallelOperations, initialize_parallel_env

class ParallelTrainer:
    """Main trainer class for distributed training with model and data parallelism."""
    
    def __init__(
        self,
        config: ModelParallelConfig,
        comm: Optional[MPI.Comm] = None
    ):
        """Initialize the parallel trainer.
        
        Args:
            config: Configuration for parallel training
            comm: MPI communicator (defaults to MPI.COMM_WORLD if not provided)
        """
        self.config = config
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        
        # Initialize parallel environment
        self._initialize_parallel_env()
        
    def _initialize_parallel_env(self):
        """Initialize the parallel environment and set up communicators."""
        if not self.config.in_dim or not self.config.out_dim:
            raise ValueError("in_dim and out_dim must be set in config")
            
        results = initialize_parallel_env(
            self.comm,
            self.rank,
            self.config.mp_size,
            self.config.dp_size,
            True,  # is_fc1 for first layer
            self.config.is_megatron,
            self.config.in_dim,
            self.config.out_dim
        )
        
        (
            self.mp_idx,
            self.dp_idx,
            self.mp_comm,
            self.dp_comm,
            self.part_in_dim,
            self.part_out_dim
        ) = results
        
        self.ops = ParallelOperations()
        
    def forward(self, x: np.ndarray, is_fc1: bool = True) -> np.ndarray:
        """Perform forward pass with parallel operations.
        
        Args:
            x: Input tensor
            is_fc1: Whether this is the first FC layer
            
        Returns:
            Output tensor after parallel operations
        """
        if self.config.is_megatron:
            if is_fc1:
                return self.ops.megatron_forward_input(x, self.mp_comm, self.config.mp_size)
            else:
                return self.ops.megatron_forward_output(x, self.mp_comm, self.config.mp_size)
        else:
            if is_fc1:
                return self.ops.naive_forward_input(x, self.mp_comm, self.config.mp_size)
            else:
                return self.ops.naive_forward_output(x, self.mp_comm, self.config.mp_size)
    
    def backward(self, grad: np.ndarray, is_fc1: bool = True) -> np.ndarray:
        """Perform backward pass with parallel operations.
        
        Args:
            grad: Gradient tensor
            is_fc1: Whether this is the first FC layer
            
        Returns:
            Processed gradient tensor
        """
        if self.config.is_megatron:
            if is_fc1:
                return self.ops.megatron_backward_output(grad, self.mp_idx, self.config.mp_size)
            else:
                return self.ops.megatron_backward_x(grad, self.mp_comm, self.config.mp_size)
        else:
            if is_fc1:
                return self.ops.naive_backward_output(grad, self.mp_idx, self.config.mp_size)
            else:
                return self.ops.naive_backward_x(grad, self.mp_comm, self.config.mp_size) 