import numpy as np
from mpi4py import MPI
from typing import Dict, List, Optional, Tuple, Union, Callable
from enum import Enum

class ZeROStage(Enum):
    """Enum for ZeRO optimization stages."""
    STAGE_1 = 1  # Optimizer State Partitioning
    STAGE_2 = 2  # Gradient Partitioning + Stage 1
    STAGE_3 = 3  # Parameter Partitioning + Stage 2

class ZeROOptimizer:
    """
    Zero Redundancy Optimizer (ZeRO) implementation.
    
    ZeRO is a memory optimization technique that partitions optimizer states, 
    gradients, and model parameters across data parallel processes to reduce 
    memory redundancy.
    
    Stages:
    1. Optimizer State Partitioning: Each process stores only a subset of optimizer states
    2. Gradient Partitioning: Each process stores only a subset of gradients
    3. Parameter Partitioning: Each process stores only a subset of parameters
    
    References:
    - ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
      (https://arxiv.org/abs/1910.02054)
    """
    
    def __init__(self, dp_comm, stage=ZeROStage.STAGE_1, learning_rate=0.01, 
                 weight_decay=0.0, momentum=0.9, epsilon=1e-8):
        """
        Initialize the ZeRO optimizer.
        
        Args:
            dp_comm: MPI communicator for data parallelism
            stage: ZeRO stage (1, 2, or 3)
            learning_rate: Learning rate
            weight_decay: Weight decay coefficient
            momentum: Momentum coefficient
            epsilon: Small value for numerical stability
        """
        self.dp_comm = dp_comm
        self.stage = stage
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.epsilon = epsilon
        
        self.rank = dp_comm.Get_rank()
        self.world_size = dp_comm.Get_size()
        
        # Mapping from parameter name to partition ID
        self.param_to_partition = {}
        # Optimizer states
        self.momentum_buffers = {}
        # For tracking parameter shapes to reconstruct partitioned parameters
        self.param_shapes = {}
        
        self.param_names = []
        self.partitioned_params = {}
    
    def register_parameters(self, params):
        """
        Register parameters with the optimizer and initialize optimizer states.
        
        Args:
            params: Dictionary of parameters {name: parameter}
        """
        self.param_names = list(params.keys())
        self.param_shapes = {name: param.shape for name, param in params.items()}
        
        # Assign each parameter to a partition
        for i, name in enumerate(self.param_names):
            partition_id = i % self.world_size
            self.param_to_partition[name] = partition_id
            
            # Initialize optimizer states if this parameter belongs to this partition
            if partition_id == self.rank:
                self.momentum_buffers[name] = np.zeros_like(params[name])
                
                # For stage 3, keep a copy of the partitioned parameters
                if self.stage == ZeROStage.STAGE_3:
                    self.partitioned_params[name] = params[name].copy()
    
    def _get_partition_for_param(self, param_name):
        """Get the partition ID for a parameter."""
        return self.param_to_partition.get(param_name, None)
    
    def reduce_gradients(self, gradients):
        """
        Reduce gradients across data parallel processes according to ZeRO stage.
        
        Args:
            gradients: Dictionary of gradients {name: gradient}
            
        Returns:
            Dictionary of reduced gradients (partitioned according to ZeRO stage)
        """
        reduced_grads = {}
        
        # Stage 1: All processes have all gradients
        if self.stage == ZeROStage.STAGE_1:
            for name, grad in gradients.items():
                # Allreduce: each process gets the sum of gradients across all processes
                reduced = np.zeros_like(grad)
                self.dp_comm.Allreduce(grad, reduced, op=MPI.SUM)
                # Compute average
                reduced_grads[name] = reduced / self.world_size
        
        # Stage 2/3: Gradients are partitioned
        else:
            for name, grad in gradients.items():
                partition_id = self.param_to_partition[name]
                
                # Reduce gradients to the owner process
                reduced = np.zeros_like(grad)
                self.dp_comm.Reduce(grad, reduced, op=MPI.SUM, root=partition_id)
                
                # Only the owner process keeps the reduced gradient
                if partition_id == self.rank:
                    reduced_grads[name] = reduced / self.world_size
        
        return reduced_grads
    
    def step(self, parameters, gradients):
        """
        Perform an optimization step.
        
        Args:
            parameters: Dictionary of parameters {name: parameter}
            gradients: Dictionary of gradients {name: gradient}
            
        Returns:
            Dictionary of updated parameters
        """
        updated_params = parameters.copy()
        
        # Perform optimization step on parameters owned by this process
        for name, grad in gradients.items():
            if self.param_to_partition[name] == self.rank:
                param = parameters[name]
                
                # Apply weight decay
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * param
                
                # Update momentum buffer (m = momentum * m + grad)
                self.momentum_buffers[name] = (
                    self.momentum * self.momentum_buffers[name] + grad
                )
                
                # Update parameter (p = p - lr * m)
                updated = param - self.learning_rate * self.momentum_buffers[name]
                
                # Stage 3: Keep updated parameters locally
                if self.stage == ZeROStage.STAGE_3:
                    self.partitioned_params[name] = updated
                else:
                    updated_params[name] = updated
        
        # Stage 1 & 2: All processes broadcast their updated parameters
        if self.stage != ZeROStage.STAGE_3:
            for name in self.param_names:
                partition_id = self.param_to_partition[name]
                
                # Get shape for broadcasting
                shape = self.param_shapes[name]
                size = np.prod(shape).astype(int)
                
                # Create send and receive buffers of the same size
                buffer = np.zeros(size, dtype=np.float32)
                
                # If this rank owns the parameter, pack it into the buffer
                if partition_id == self.rank:
                    buffer[:] = updated_params[name].reshape(-1)
                
                # Broadcast from the owner rank
                self.dp_comm.Bcast(buffer, root=partition_id)
                
                # If this is not the owner, update the parameter with the received data
                if partition_id != self.rank:
                    updated_params[name] = buffer.reshape(shape)
        
        # Stage 3: All processes need to gather all parameters
        else:
            for name in self.param_names:
                partition_id = self.param_to_partition[name]
                
                # Get shape for broadcasting
                shape = self.param_shapes[name]
                size = np.prod(shape).astype(int)
                
                # Create buffer for broadcasting
                buffer = np.zeros(size, dtype=np.float32)
                
                # If this rank owns the parameter, pack it into the buffer
                if partition_id == self.rank:
                    buffer[:] = self.partitioned_params[name].reshape(-1)
                
                # Broadcast from the owner rank
                self.dp_comm.Bcast(buffer, root=partition_id)
                
                # Update param with the received data
                updated_params[name] = buffer.reshape(shape)
        
        return updated_params 