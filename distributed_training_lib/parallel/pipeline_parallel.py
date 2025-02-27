import numpy as np
from mpi4py import MPI
from typing import List, Dict, Any, Optional, Tuple, Union
import time

class PipelineStage:
    """A single stage in a pipeline parallel setup."""
    
    def __init__(
        self, 
        stage_id: int,
        num_stages: int,
        micro_batch_size: int,
        comm: MPI.Comm,
        world_rank: int,
        next_stage_rank: Optional[int] = None,
        prev_stage_rank: Optional[int] = None
    ):
        """Initialize pipeline stage.
        
        Args:
            stage_id: ID of this stage (0-indexed)
            num_stages: Total number of stages in the pipeline
            micro_batch_size: Size of micro-batches
            comm: MPI communicator
            world_rank: Global rank of this process
            next_stage_rank: World rank of the next stage (for sending)
            prev_stage_rank: World rank of the previous stage (for receiving)
        """
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.micro_batch_size = micro_batch_size
        self.comm = comm
        self.world_rank = world_rank
        
        # Calculate next and previous stage ranks if not provided
        self.next_stage_rank = next_stage_rank
        self.prev_stage_rank = prev_stage_rank
        
        # Buffer for activations and gradients
        self.activation_buffer = {}
        self.gradient_buffer = {}
        
    def is_first_stage(self) -> bool:
        """Check if this is the first stage in the pipeline."""
        return self.stage_id == 0
    
    def is_last_stage(self) -> bool:
        """Check if this is the last stage in the pipeline."""
        return self.stage_id == self.num_stages - 1
    
    def forward(self, micro_batch_id: int, input_data: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Forward pass for a micro-batch.
        
        Args:
            micro_batch_id: ID of the micro-batch
            input_data: Input data for this stage
            
        Returns:
            Output activations or None if not last stage
        """
        # If not first stage, receive input from previous stage
        if not self.is_first_stage() and input_data is None:
            if self.prev_stage_rank is None:
                raise ValueError("Previous stage rank is required for non-first stages")
                
            status = MPI.Status()
            # For simplicity, we'll use a fixed shape for communication
            input_data = np.empty((self.micro_batch_size, 4), dtype=np.float32)
            self.comm.Recv(
                [input_data, MPI.FLOAT],
                source=self.prev_stage_rank,
                tag=micro_batch_id,
                status=status
            )
        
        # Process the input (this would be replaced with actual model computation)
        # For testing, we'll just add the stage ID to the input
        if input_data is not None:
            output_data = input_data + self.stage_id
        else:
            # If no input (shouldn't happen), create dummy data
            output_data = np.ones((self.micro_batch_size, 4)) * self.stage_id
        
        # Store activations for backward pass
        self.activation_buffer[micro_batch_id] = output_data.copy()
        
        # If not last stage, send to next stage
        if not self.is_last_stage():
            if self.next_stage_rank is None:
                raise ValueError("Next stage rank is required for non-last stages")
                
            self.comm.Send(
                [output_data, MPI.FLOAT],
                dest=self.next_stage_rank,
                tag=micro_batch_id
            )
            return None
        else:
            return output_data
    
    def backward(self, micro_batch_id: int, output_gradients: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Backward pass for a micro-batch.
        
        Args:
            micro_batch_id: ID of the micro-batch
            output_gradients: Gradients of the loss with respect to output
            
        Returns:
            Input gradients or None if first stage
        """
        # If not last stage, receive gradients from next stage
        if not self.is_last_stage() and output_gradients is None:
            if self.next_stage_rank is None:
                raise ValueError("Next stage rank is required for non-last stages")
                
            status = MPI.Status()
            # For simplicity, we'll use a fixed shape for communication
            output_gradients = np.empty((self.micro_batch_size, 4), dtype=np.float32)
            self.comm.Recv(
                [output_gradients, MPI.FLOAT],
                source=self.next_stage_rank,
                tag=micro_batch_id + 1000,  # Offset tags for backward pass
                status=status
            )
        
        # Get saved activations
        activations = self.activation_buffer.get(micro_batch_id, None)
        if activations is None:
            raise ValueError(f"No activations found for micro-batch {micro_batch_id}")
        
        # Process gradients (this would be replaced with actual gradient computation)
        # For testing, we'll just multiply gradients by stage ID + 1
        if output_gradients is not None:
            input_gradients = output_gradients * (self.stage_id + 1)
        else:
            # If no gradients (shouldn't happen), create dummy data
            input_gradients = np.ones_like(activations) * (self.stage_id + 1)
        
        # Store gradients
        self.gradient_buffer[micro_batch_id] = input_gradients.copy()
        
        # If not first stage, send gradients to previous stage
        if not self.is_first_stage():
            if self.prev_stage_rank is None:
                raise ValueError("Previous stage rank is required for non-first stages")
                
            self.comm.Send(
                [input_gradients, MPI.FLOAT],
                dest=self.prev_stage_rank,
                tag=micro_batch_id + 1000  # Offset tags for backward pass
            )
            return None
        else:
            return input_gradients
    
    def clear_buffers(self):
        """Clear activation and gradient buffers."""
        self.activation_buffer.clear()
        self.gradient_buffer.clear()

class PipelineScheduler:
    """Scheduler for pipeline parallel training."""
    
    def __init__(
        self,
        num_stages: int,
        num_micro_batches: int,
        stage: PipelineStage
    ):
        """Initialize pipeline scheduler.
        
        Args:
            num_stages: Total number of stages in pipeline
            num_micro_batches: Number of micro-batches per batch
            stage: Pipeline stage this scheduler controls
        """
        self.num_stages = num_stages
        self.num_micro_batches = num_micro_batches
        self.stage = stage
        
    def schedule_1f1b(self, input_micro_batches: Optional[List[np.ndarray]] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Run 1F1B (one-forward-one-backward) schedule.
        
        This alternates between forward and backward passes to maximize pipeline utilization.
        
        Args:
            input_micro_batches: List of input micro-batches for first stage
            
        Returns:
            Tuple of (outputs, gradients) from pipeline
        """
        outputs = []
        gradients = []
        
        # Clear stage buffers
        self.stage.clear_buffers()
        
        # Warmup phase - forward passes
        for micro_batch_id in range(min(self.num_stages - 1, self.num_micro_batches)):
            input_data = None
            if self.stage.is_first_stage() and input_micro_batches is not None:
                input_data = input_micro_batches[micro_batch_id]
                
            output = self.stage.forward(micro_batch_id, input_data)
            if output is not None:
                outputs.append(output)
        
        # 1F1B steady state phase
        for micro_batch_id in range(self.num_micro_batches - (self.num_stages - 1)):
            # Forward pass
            forward_id = micro_batch_id + self.num_stages - 1
            input_data = None
            if self.stage.is_first_stage() and input_micro_batches is not None:
                input_data = input_micro_batches[forward_id]
                
            output = self.stage.forward(forward_id, input_data)
            if output is not None:
                outputs.append(output)
            
            # Backward pass
            backward_id = micro_batch_id
            output_gradients = None
            if self.stage.is_last_stage():
                # For testing, use ones as loss gradients
                output_gradients = np.ones_like(self.stage.activation_buffer[backward_id])
                
            gradient = self.stage.backward(backward_id, output_gradients)
            if gradient is not None:
                gradients.append(gradient)
        
        # Cooldown phase - backward passes
        for micro_batch_id in range(self.num_micro_batches - (self.num_stages - 1), self.num_micro_batches):
            output_gradients = None
            if self.stage.is_last_stage():
                # For testing, use ones as loss gradients
                output_gradients = np.ones_like(self.stage.activation_buffer[micro_batch_id])
                
            gradient = self.stage.backward(micro_batch_id, output_gradients)
            if gradient is not None:
                gradients.append(gradient)
        
        return outputs, gradients

class PipelineParallelConfig:
    """Configuration for pipeline parallel training."""
    
    def __init__(
        self,
        num_stages: int,
        micro_batch_size: int,
        num_micro_batches: int,
        dp_size: int = 1
    ):
        """Initialize pipeline parallel configuration.
        
        Args:
            num_stages: Number of pipeline stages
            micro_batch_size: Size of each micro-batch
            num_micro_batches: Number of micro-batches per batch
            dp_size: Data parallel size (default: 1)
        """
        self.num_stages = num_stages
        self.micro_batch_size = micro_batch_size
        self.num_micro_batches = num_micro_batches
        self.dp_size = dp_size
        self.batch_size = micro_batch_size * num_micro_batches 