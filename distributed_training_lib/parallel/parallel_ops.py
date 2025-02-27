import numpy as np
from mpi4py import MPI
from typing import Tuple, Optional

def initialize_parallel_env(
    comm,
    rank: int,
    mp_size: int,
    dp_size: int,
    is_fc1: bool,
    is_megatron_mp: bool,
    in_dim: int,
    out_dim: int,
) -> Tuple[int, int, MPI.Comm, MPI.Comm, int, int]:
    """Initialize the parallel environment and return necessary information.

    Args:
        comm: Global MPI communicator
        rank: Process rank
        mp_size: Model parallel size
        dp_size: Data parallel size
        is_fc1: Whether this is the first FC layer
        is_megatron_mp: Whether to use Megatron-style model parallelism
        in_dim: Input dimension
        out_dim: Output dimension

    Returns:
        Tuple containing:
        - mp_idx: Model parallel index
        - dp_idx: Data parallel index
        - mp_comm: Model parallel communicator
        - dp_comm: Data parallel communicator
        - part_in_dim: Partitioned input dimension
        - part_out_dim: Partitioned output dimension
    """
    mp_idx = rank % mp_size
    dp_idx = rank // mp_size
    
    mp_comm = comm.Split(color=dp_idx, key=mp_idx)
    dp_comm = comm.Split(color=mp_idx, key=dp_idx)
    
    if is_megatron_mp:
        if is_fc1:
            part_in_dim = in_dim
            part_out_dim = out_dim // mp_size
        else:
            part_in_dim = in_dim // mp_size
            part_out_dim = out_dim
    else:
        part_in_dim = in_dim
        part_out_dim = out_dim // mp_size
    
    return mp_idx, dp_idx, mp_comm, dp_comm, part_in_dim, part_out_dim


class ParallelOperations:
    """Class containing parallel operations for both naive and Megatron-style model parallelism."""
    
    @staticmethod
    def naive_forward_input(x: np.ndarray, mp_comm: MPI.Comm, mp_size: int) -> np.ndarray:
        """Collect forward inputs for naive model parallelism."""
        gathered = mp_comm.allgather(x)
        return np.concatenate(gathered, axis=1)
    
    @staticmethod
    def naive_forward_output(out: np.ndarray, mp_comm: MPI.Comm, mp_size: int) -> np.ndarray:
        """Collect forward outputs for naive model parallelism."""
        gathered = mp_comm.allgather(out)
        return np.concatenate(gathered, axis=1)
    
    @staticmethod
    def megatron_forward_input(x: np.ndarray, mp_comm: MPI.Comm, mp_size: int) -> np.ndarray:
        """Collect forward inputs for Megatron-style model parallelism."""
        return x
    
    @staticmethod
    def megatron_forward_output(out: np.ndarray, mp_comm: MPI.Comm, mp_size: int) -> np.ndarray:
        """Collect forward outputs for Megatron-style model parallelism."""
        collected_out = np.empty_like(out)
        mp_comm.Allreduce(out, collected_out, op=MPI.SUM)
        return collected_out
    
    @staticmethod
    def naive_backward_output(output_grad: np.ndarray, mp_group_idx: int, mp_size: int) -> np.ndarray:
        """Collect backward outputs for naive model parallelism."""
        splits = np.array_split(output_grad, mp_size, axis=1)
        return splits[mp_group_idx]
    
    @staticmethod
    def naive_backward_x(grad_x: np.ndarray, mp_comm: MPI.Comm, mp_size: int) -> np.ndarray:
        """Collect backward gradients for naive model parallelism."""
        collected_grad_x = np.empty_like(grad_x)
        mp_comm.Allreduce(grad_x, collected_grad_x, op=MPI.SUM)
        return collected_grad_x
    
    @staticmethod
    def megatron_backward_output(output_grad: np.ndarray, mp_group_idx: int, mp_size: int) -> np.ndarray:
        """Collect backward outputs for Megatron-style model parallelism."""
        return output_grad
    
    @staticmethod
    def megatron_backward_x(grad_x: np.ndarray, mp_comm: MPI.Comm, mp_size: int) -> np.ndarray:
        """Collect backward gradients for Megatron-style model parallelism."""
        gathered = mp_comm.allgather(grad_x)
        return np.concatenate(gathered, axis=1) 