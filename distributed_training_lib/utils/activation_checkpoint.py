import numpy as np
import time
import logging
from typing import Dict, List, Callable, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class ActivationCheckpoint:
    """
    Implementation of activation checkpointing for memory optimization.
    
    Activation checkpointing (or gradient checkpointing) is a technique to reduce 
    memory usage during training by not storing all activations for the backward pass.
    Instead, activations are recomputed during the backward pass as needed.
    
    This implementation is framework-agnostic and works with pure NumPy.
    
    References:
    - Memory-Efficient Backpropagation Through Time
      (https://arxiv.org/abs/1606.03401)
    """
    
    def __init__(self, checkpoint_segments=1, debug=False):
        """
        Initialize activation checkpointing.
        
        Args:
            checkpoint_segments: Number of segments to divide the network into.
                                 Higher values save more memory but require more recomputation.
            debug: Enable debug logging
        """
        self.checkpoint_segments = max(1, checkpoint_segments)
        self.debug = debug
        self.checkpointed_activations = {}
        self.function_cache = {}
        
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
    
    def register_function(self, name: str, forward_fn: Callable, backward_fn: Callable):
        """
        Register a function for checkpointing.
        
        Args:
            name: Name of the function
            forward_fn: Forward pass function (inputs -> outputs)
            backward_fn: Backward pass function (inputs, outputs, grad_outputs -> grad_inputs)
        """
        self.function_cache[name] = {
            "forward": forward_fn,
            "backward": backward_fn
        }
        logger.debug(f"Registered function '{name}' for checkpointing")
    
    def _should_checkpoint(self, layer_idx: int, total_layers: int) -> bool:
        """
        Determine if a layer should be checkpointed.
        
        Args:
            layer_idx: Index of the current layer
            total_layers: Total number of layers
            
        Returns:
            True if the layer should be checkpointed
        """
        # Simple strategy: checkpoint boundary layers between segments
        segment_size = max(1, total_layers // self.checkpoint_segments)
        return (layer_idx % segment_size == 0)
    
    def checkpoint_activations(self, name: str, layer_idx: int, total_layers: int, 
                             inputs: List[np.ndarray], rng_state=None) -> Tuple[List[np.ndarray], Any]:
        """
        Forward pass with activation checkpointing.
        
        Args:
            name: Name of the registered function
            layer_idx: Index of the current layer
            total_layers: Total number of layers
            inputs: List of input tensors
            rng_state: Optional random state to save
            
        Returns:
            Tuple of (outputs, None or saved_tensors)
        """
        if name not in self.function_cache:
            raise ValueError(f"Function '{name}' not registered for checkpointing")
        
        forward_fn = self.function_cache[name]["forward"]
        
        # Check if this is a checkpoint boundary
        is_checkpoint = self._should_checkpoint(layer_idx, total_layers)
        
        # Get outputs from forward function
        outputs = forward_fn(*inputs)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        
        # Save activations if this is a checkpoint boundary
        if is_checkpoint:
            saved_tensors = {
                "inputs": [x.copy() for x in inputs],
                "outputs": [x.copy() for x in outputs],
                "rng_state": rng_state
            }
            self.checkpointed_activations[layer_idx] = saved_tensors
            logger.debug(f"Checkpointed layer {layer_idx}/{total_layers}")
        
        return outputs
    
    def backward_pass(self, name: str, layer_idx: int, total_layers: int, 
                    grad_outputs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Backward pass with activation checkpointing.
        
        Args:
            name: Name of the registered function
            layer_idx: Index of the current layer
            total_layers: Total number of layers
            grad_outputs: List of gradient tensors for outputs
            
        Returns:
            List of gradient tensors for inputs
        """
        if name not in self.function_cache:
            raise ValueError(f"Function '{name}' not registered for checkpointing")
        
        forward_fn = self.function_cache[name]["forward"]
        backward_fn = self.function_cache[name]["backward"]
        
        # Check if this is a checkpoint boundary (saved activations)
        if layer_idx in self.checkpointed_activations:
            saved_tensors = self.checkpointed_activations[layer_idx]
            inputs = saved_tensors["inputs"]
            outputs = saved_tensors["outputs"]
            logger.debug(f"Using checkpointed activations for layer {layer_idx}/{total_layers}")
        else:
            # Need to recompute activations for this layer
            # Find the nearest checkpoint before this layer
            checkpoint_idx = max([i for i in self.checkpointed_activations.keys() if i < layer_idx], default=None)
            
            if checkpoint_idx is None:
                raise ValueError(f"No checkpoint found before layer {layer_idx}")
            
            # Start from the checkpoint and recompute
            saved_tensors = self.checkpointed_activations[checkpoint_idx]
            current_inputs = saved_tensors["inputs"]
            
            logger.debug(f"Recomputing activations from layer {checkpoint_idx} to {layer_idx}")
            
            # Recompute all layers from checkpoint to current layer
            for i in range(checkpoint_idx, layer_idx):
                current_fn_name = f"{name}_{i}" if i != layer_idx else name
                if current_fn_name in self.function_cache:
                    current_forward_fn = self.function_cache[current_fn_name]["forward"]
                else:
                    current_forward_fn = self.function_cache[name]["forward"]
                
                current_outputs = current_forward_fn(*current_inputs)
                if not isinstance(current_outputs, (list, tuple)):
                    current_outputs = [current_outputs]
                
                current_inputs = current_outputs
            
            inputs = current_inputs
            outputs = forward_fn(*inputs)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
        
        # Compute gradients
        if not isinstance(grad_outputs, (list, tuple)):
            grad_outputs = [grad_outputs]
        
        grad_inputs = backward_fn(inputs, outputs, grad_outputs)
        if not isinstance(grad_inputs, (list, tuple)):
            grad_inputs = [grad_inputs]
        
        return grad_inputs
    
    def clear_checkpoints(self):
        """
        Clear all checkpointed activations.
        """
        self.checkpointed_activations.clear()
        logger.debug("Cleared all checkpointed activations")

class CheckpointFunction:
    """
    A wrapper function that applies activation checkpointing.
    This provides a simpler interface for checkpointing specific functions.
    """
    
    @staticmethod
    def apply(function: Callable, *args, preserve_rng_state=True):
        """
        Apply checkpointing to a function.
        
        Args:
            function: Function to checkpoint (inputs -> outputs)
            *args: Function inputs
            preserve_rng_state: Whether to preserve random state
            
        Returns:
            Function outputs
        """
        # Save inputs
        saved_args = [arg.copy() if isinstance(arg, np.ndarray) else arg for arg in args]
        
        # Save RNG state if needed
        if preserve_rng_state:
            rng_state = np.random.get_state()
        
        # Run forward pass
        with torch_disable_grad():
            outputs = function(*args)
        
        # Define backward function for checkpointing
        def backward_function(*grad_outputs):
            # Restore RNG state if needed
            if preserve_rng_state:
                np.random.set_state(rng_state)
            
            # Enable grad and recompute
            return checkpoint_backward(*saved_args, function, grad_outputs)
        
        # Return outputs with gradient function attached
        return CustomGradientFunction.apply(outputs, backward_function)

def torch_disable_grad():
    """Context manager to disable gradient calculation."""
    class DummyContextManager:
        def __enter__(self):
            pass
        
        def __exit__(self, *args):
            pass
    
    return DummyContextManager()

def checkpoint_backward(function, *args, grad_outputs):
    """
    Backward pass for checkpointed function.
    
    Args:
        function: Function that was checkpointed
        *args: Function inputs
        grad_outputs: Gradients of outputs
        
    Returns:
        Gradients of inputs
    """
    # Recompute forward pass
    with torch_enable_grad():
        inputs = [arg.copy() if isinstance(arg, np.ndarray) else arg for arg in args[:-2]]
        outputs = function(*inputs)
        
        # Compute gradients
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        if not isinstance(grad_outputs, (list, tuple)):
            grad_outputs = [grad_outputs]
        
        # Manual gradient computation (simplified)
        grad_inputs = []
        for inp in inputs:
            if isinstance(inp, np.ndarray):
                grad_inputs.append(np.zeros_like(inp))
            else:
                grad_inputs.append(None)
        
        # This is a simplification - in a real implementation, 
        # you would compute proper gradients here
        
        return grad_inputs

def torch_enable_grad():
    """Context manager to enable gradient calculation."""
    return torch_disable_grad()

class CustomGradientFunction:
    """
    Simple class to mimic PyTorch's custom gradient function.
    For a real implementation, this would be replaced with
    framework-specific autograd functions.
    """
    
    @staticmethod
    def apply(outputs, backward_fn):
        """Apply custom gradient."""
        # In a real implementation, this would hook into autograd
        return outputs 