import numpy as np
from mpi4py import MPI
from typing import List, Dict, Any, Optional, Tuple, Union
import time

from ..core.trainer import ParallelTrainer
from .pipeline_parallel import (
    PipelineStage, 
    PipelineScheduler, 
    PipelineParallelConfig
)

class PipelineParallelTrainer:
    """Trainer for pipeline parallel distributed training."""
    
    def __init__(
        self,
        config: PipelineParallelConfig,
        world_comm: Optional[MPI.Comm] = None
    ):
        """Initialize pipeline parallel trainer.
        
        Args:
            config: Pipeline parallel configuration
            world_comm: World communicator (default: MPI.COMM_WORLD)
        """
        # MPI setup
        self.world_comm = world_comm if world_comm else MPI.COMM_WORLD
        self.world_size = self.world_comm.Get_size()
        self.world_rank = self.world_comm.Get_rank()
        
        # Configuration
        self.config = config
        
        # Validate configuration
        total_procs_needed = config.num_stages * config.dp_size
        if self.world_size < total_procs_needed:
            raise ValueError(
                f"Not enough processes for configuration. "
                f"Need {total_procs_needed}, but only {self.world_size} available."
            )
        
        # Create communicators
        self._create_communicators()
        
        # Determine stage and data parallel ranks
        self.pipeline_stage_rank = self.world_rank % self.config.num_stages
        self.dp_rank = self.world_rank // self.config.num_stages
        
        # Determine adjacent stage ranks (for communication)
        self.next_stage_rank = None
        self.prev_stage_rank = None
        
        if not self.is_last_stage():
            # Next stage has rank = current_rank + 1 (in the same dp group)
            self.next_stage_rank = self.world_rank + 1
        
        if not self.is_first_stage():
            # Previous stage has rank = current_rank - 1 (in the same dp group)
            self.prev_stage_rank = self.world_rank - 1
        
        # Initialize pipeline stage
        self.stage = PipelineStage(
            stage_id=self.pipeline_stage_rank,
            num_stages=config.num_stages,
            micro_batch_size=config.micro_batch_size,
            comm=self.world_comm,
            world_rank=self.world_rank,
            next_stage_rank=self.next_stage_rank,
            prev_stage_rank=self.prev_stage_rank
        )
        
        # Initialize scheduler
        self.scheduler = PipelineScheduler(
            num_stages=config.num_stages,
            num_micro_batches=config.num_micro_batches,
            stage=self.stage
        )
        
        # Print initialization info
        if self.world_rank == 0:
            print(f"Initialized PipelineParallelTrainer with:")
            print(f"  Pipeline Stages: {config.num_stages}")
            print(f"  Data Parallel Size: {config.dp_size}")
            print(f"  Micro-batch Size: {config.micro_batch_size}")
            print(f"  Num Micro-batches: {config.num_micro_batches}")
            print(f"  Total Batch Size: {config.batch_size}")
    
    def is_first_stage(self) -> bool:
        """Check if this is the first stage in the pipeline."""
        return self.pipeline_stage_rank == 0
    
    def is_last_stage(self) -> bool:
        """Check if this is the last stage in the pipeline."""
        return self.pipeline_stage_rank == self.config.num_stages - 1
    
    def _create_communicators(self):
        """Create MPI communicators for pipeline and data parallelism."""
        # Create pipeline stage communicator
        # This groups processes by their stage rank (for DP within a stage)
        color_pipeline = self.world_rank % self.config.num_stages
        key_pipeline = self.world_rank // self.config.num_stages
        self.pipeline_stage_comm = self.world_comm.Split(color_pipeline, key_pipeline)
        
        # Create data parallel communicator
        # This groups processes by their DP rank (across stages)
        color_dp = self.world_rank // self.config.num_stages
        key_dp = self.world_rank % self.config.num_stages
        self.dp_comm = self.world_comm.Split(color_dp, key_dp)
    
    def train_batch(self, input_batches: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Train a single batch using pipeline parallelism.
        
        Args:
            input_batches: Input data for first stage
                Should have shape (micro_batch_size * num_micro_batches, input_dim)
            
        Returns:
            Tuple of (outputs, gradients) for the batch
        """
        # Only first stage needs input data
        micro_batches = None
        if self.is_first_stage() and input_batches is not None:
            # Split input into micro-batches
            num_samples = input_batches.shape[0]
            if num_samples != self.config.batch_size:
                raise ValueError(
                    f"Input batch size {num_samples} doesn't match "
                    f"expected size {self.config.batch_size}"
                )
            
            micro_batches = []
            for i in range(self.config.num_micro_batches):
                start_idx = i * self.config.micro_batch_size
                end_idx = start_idx + self.config.micro_batch_size
                micro_batches.append(input_batches[start_idx:end_idx])
        
        # Run 1F1B schedule
        outputs, gradients = self.scheduler.schedule_1f1b(micro_batches)
        
        return outputs, gradients
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass using pipeline parallelism.
        
        This is a simpler interface that doesn't return intermediate results.
        
        Args:
            input_data: Input data
            
        Returns:
            Output from last stage
        """
        outputs, _ = self.train_batch(input_data)
        
        # Only last stage has outputs
        if self.is_last_stage():
            # Concatenate outputs from micro-batches
            if outputs:
                return np.vstack(outputs)
            else:
                return np.array([])
        else:
            return np.array([])
    
    def backward(self, output_gradients: np.ndarray) -> np.ndarray:
        """Backward pass using pipeline parallelism.
        
        Args:
            output_gradients: Gradients of loss with respect to output
            
        Returns:
            Gradients from first stage
        """
        # Not implemented as separate call - use train_batch instead
        # This is a placeholder for API compatibility
        return np.array([])
        
    def cleanup(self):
        """Clean up resources."""
        self.stage.clear_buffers() 