import numpy as np
from mpi4py import MPI
import math

class TensorParallelism:
    """
    Implementation of tensor parallelism for distributed training.
    
    Tensor parallelism splits individual tensors (typically weight matrices) 
    across multiple devices, allowing for training of larger models than 
    would fit on a single device. This is particularly useful for large
    Transformer models.
    
    This implementation focuses on splitting linear layers, which are the 
    most common bottleneck in large models.
    
    References:
    - Megatron-LM: Training Multi-Billion Parameter Language Models Using
      Model Parallelism (https://arxiv.org/abs/1909.08053)
    """
    
    def __init__(self, comm, split_dim=0):
        """
        Initialize tensor parallelism.
        
        Args:
            comm: MPI communicator for tensor parallelism
            split_dim: Dimension along which to split tensors (0 for row, 1 for column)
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.world_size = comm.Get_size()
        self.split_dim = split_dim
    
    def split_tensor(self, tensor, name=None):
        """
        Split a tensor along the specified dimension.
        
        Args:
            tensor: NumPy array to split
            name: Optional name for the tensor for debugging
            
        Returns:
            Local portion of the tensor for this rank
        """
        shape = tensor.shape
        
        if len(shape) < 2:
            # Can't split vectors or scalars
            return tensor
        
        # Calculate split size and leftover elements
        split_size = shape[self.split_dim] // self.world_size
        remainder = shape[self.split_dim] % self.world_size
        
        # Adjust split size for this rank if necessary
        local_split_size = split_size
        start_idx = self.rank * split_size
        
        if self.rank < remainder:
            local_split_size += 1
            start_idx += self.rank
        else:
            start_idx += remainder
        
        # Create slice for this rank
        slice_indices = [slice(None)] * len(shape)
        slice_indices[self.split_dim] = slice(start_idx, start_idx + local_split_size)
        
        local_tensor = tensor[tuple(slice_indices)].copy()
        
        if name:
            print(f"Rank {self.rank}: Split tensor '{name}' with shape {shape} -> {local_tensor.shape}")
            
        return local_tensor
    
    def gather_tensor(self, local_tensor, name=None):
        """
        Gather a tensor that was previously split.
        
        Args:
            local_tensor: Local portion of the tensor
            name: Optional name for the tensor for debugging
            
        Returns:
            Complete tensor gathered from all ranks
        """
        local_shape = local_tensor.shape
        
        # Get sizes from all ranks
        local_size = np.array([local_shape[self.split_dim]], dtype=np.int32)
        all_sizes = np.zeros(self.world_size, dtype=np.int32)
        
        self.comm.Allgather(local_size, all_sizes)
        
        # Calculate complete tensor shape
        full_shape = list(local_shape)
        full_shape[self.split_dim] = np.sum(all_sizes)
        
        # Create buffer for the full tensor
        full_tensor = np.zeros(full_shape, dtype=local_tensor.dtype)
        
        # Create slice for this rank's contribution
        start_idx = sum(all_sizes[:self.rank])
        slice_indices = [slice(None)] * len(full_shape)
        slice_indices[self.split_dim] = slice(start_idx, start_idx + local_shape[self.split_dim])
        
        # Place local tensor in the full tensor
        full_tensor[tuple(slice_indices)] = local_tensor
        
        # Gather all contributions
        self.comm.Allreduce(MPI.IN_PLACE, full_tensor, op=MPI.SUM)
        
        if name:
            print(f"Rank {self.rank}: Gathered tensor '{name}' with shape {full_shape}")
            
        return full_tensor
    
    def split_linear_layer(self, weights, biases=None, split_type="column"):
        """
        Split a linear layer across devices.
        
        Args:
            weights: Weight matrix (2D numpy array)
            biases: Optional bias vector
            split_type: How to split the layer ('column' or 'row')
            
        Returns:
            Tuple of (local_weights, local_biases)
        """
        if split_type not in ["column", "row"]:
            raise ValueError("split_type must be 'column' or 'row'")
        
        # Set split dimension based on split type
        split_dim = 1 if split_type == "column" else 0
        
        # Split weights
        local_weights = self.split_tensor(weights, name=f"weights_{split_type}_split")
        
        # Handle biases
        local_biases = None
        if biases is not None:
            if split_type == "column":
                # For column-wise split, each worker gets the full bias vector
                local_biases = biases
            else:
                # For row-wise split, split the bias vector
                local_biases = self.split_tensor(biases, name=f"biases_{split_type}_split")
        
        return local_weights, local_biases
    
    def forward_column_parallel(self, input_tensor, weights, biases=None):
        """
        Forward pass for a column-parallel linear layer.
        
        In column-parallel linear layers, the weight matrix is split along
        the column dimension, and each worker computes part of the output.
        The results are then gathered across workers.
        
        Args:
            input_tensor: Input tensor (same on all workers)
            weights: Local portion of weight matrix
            biases: Full bias vector (or None)
            
        Returns:
            Output tensor
        """
        # Local forward computation
        local_output = np.matmul(input_tensor, weights)
        
        # Gather results across workers using Allreduce for simplicity
        # This approach works because each rank computes a partial output 
        # that doesn't overlap with other ranks
        output_size = weights.shape[1] * self.world_size
        batch_size = input_tensor.shape[0]
        
        # Create a zero array with the full output size
        full_output = np.zeros((batch_size, output_size), dtype=local_output.dtype)
        
        # Place local outputs in the appropriate slice of the full output
        start_col = weights.shape[1] * self.rank
        end_col = start_col + weights.shape[1]
        full_output[:, start_col:end_col] = local_output
        
        # Allreduce to combine partial results
        self.comm.Allreduce(MPI.IN_PLACE, full_output, op=MPI.SUM)
        
        # Add bias after gathering if provided
        if biases is not None:
            full_output = full_output + biases
        
        return full_output
    
    def forward_row_parallel(self, input_tensor, weights, biases=None):
        """
        Forward pass for a row-parallel linear layer.
        
        In row-parallel linear layers, the weight matrix is split along
        the row dimension. The input tensor is scattered, and each worker
        computes using its portion of weights. Results are then combined.
        
        Args:
            input_tensor: Input tensor (will be split across workers)
            weights: Local portion of weight matrix
            biases: Local portion of bias vector (or None)
            
        Returns:
            Output tensor
        """
        # For row-parallel, we can't just split the input directly, because
        # the dimensions won't match correctly with the weights.
        
        # Let's look at the dimensions:
        # input_tensor is (batch_size, full_in_features)
        # weights is (partial_in_features, out_features)
        
        # Calculate which rows of the weight matrix this process handles
        in_features = input_tensor.shape[1]
        rows_per_rank = in_features // self.world_size
        
        # Handle uneven division
        if self.rank < in_features % self.world_size:
            rows_per_rank += 1
            start_row = rows_per_rank * self.rank
        else:
            start_row = (rows_per_rank + 1) * (in_features % self.world_size) + rows_per_rank * (self.rank - in_features % self.world_size)
            
        end_row = start_row + weights.shape[0]
        
        # Get the corresponding slice of the input
        local_input = input_tensor[:, start_row:end_row]
        
        # Local forward computation
        local_output = np.matmul(local_input, weights)
        
        # Gather results from all processes
        output = self.comm.allreduce(local_output, op=MPI.SUM)
        
        # Add biases if provided
        if biases is not None:
            output += biases
        
        return output
    
    def backward_column_parallel(self, input_tensor, grad_output, weights):
        """
        Backward pass for a column-parallel linear layer.
        
        Args:
            input_tensor: Input tensor from forward pass
            grad_output: Gradient of the loss with respect to the output
            weights: Local portion of weight matrix
            
        Returns:
            Tuple of (grad_input, grad_weights, grad_biases)
        """
        # Each process only has a portion of the weights, and the grad_output
        # corresponds to the full output. We need to extract the part of grad_output
        # that corresponds to this process's outputs.
        
        # Calculate which columns of the output this process is responsible for
        start_col = weights.shape[1] * self.rank
        end_col = start_col + weights.shape[1]
        
        # Extract the relevant portion of grad_output
        local_grad_output = grad_output[:, start_col:end_col]
        
        # Compute gradient with respect to input (partial)
        local_grad_input = np.matmul(local_grad_output, weights.T)
        
        # Gather the full gradient for the input across all processes
        grad_input = np.zeros_like(local_grad_input)
        self.comm.Allreduce(local_grad_input, grad_input, op=MPI.SUM)
        
        # Compute gradient with respect to weights (different for each worker)
        grad_weights = np.matmul(input_tensor.T, local_grad_output)
        
        # Compute gradient with respect to biases
        grad_biases = np.sum(local_grad_output, axis=0)
        
        return grad_input, grad_weights, grad_biases
    
    def backward_row_parallel(self, input_tensor, grad_output, weights):
        """
        Backward pass for a row-parallel linear layer.
        
        Args:
            input_tensor: Input tensor from forward pass (before splitting)
            grad_output: Gradient of the loss with respect to the output
            weights: Local portion of weight matrix
            
        Returns:
            Tuple of (grad_input, grad_weights, grad_biases)
        """
        # Calculate which rows of the weight matrix this process handles
        in_features = input_tensor.shape[1]
        rows_per_rank = in_features // self.world_size
        
        # Handle uneven division
        if self.rank < in_features % self.world_size:
            rows_per_rank += 1
            start_row = rows_per_rank * self.rank
        else:
            start_row = (rows_per_rank + 1) * (in_features % self.world_size) + rows_per_rank * (self.rank - in_features % self.world_size)
            
        end_row = start_row + weights.shape[0]
        
        # Get the corresponding slice of the input
        local_input = input_tensor[:, start_row:end_row]
        
        # Compute local gradient with respect to input
        local_grad_input = np.matmul(grad_output, weights.T)
        
        # Initialize full gradient input with zeros
        grad_input = np.zeros((input_tensor.shape[0], input_tensor.shape[1]), dtype=input_tensor.dtype)
        
        # Place the local gradients in the correct position
        grad_input[:, start_row:end_row] = local_grad_input
        
        # Reduce across all processes to get complete gradients
        self.comm.Allreduce(MPI.IN_PLACE, grad_input, op=MPI.SUM)
        
        # Compute gradient with respect to weights (different for each worker)
        grad_weights = np.matmul(local_input.T, grad_output)
        
        # Compute gradient with respect to biases
        grad_biases = np.sum(grad_output, axis=0)
        
        return grad_input, grad_weights, grad_biases
    
    def parallel_linear_forward(self, input_tensor, weights, biases=None, split_type="column"):
        """
        Unified forward pass for a parallel linear layer.
        
        Args:
            input_tensor: Input tensor
            weights: Local portion of weight matrix
            biases: Local portion of bias vector (or None)
            split_type: How the layer is split ('column' or 'row')
            
        Returns:
            Output tensor
        """
        if split_type == "column":
            return self.forward_column_parallel(input_tensor, weights, biases)
        elif split_type == "row":
            return self.forward_row_parallel(input_tensor, weights, biases)
        else:
            raise ValueError("split_type must be 'column' or 'row'")
    
    def parallel_linear_backward(self, input_tensor, grad_output, weights, split_type="column"):
        """
        Unified backward pass for a parallel linear layer.
        
        Args:
            input_tensor: Input tensor from forward pass
            grad_output: Gradient of the loss with respect to the output
            weights: Local portion of weight matrix
            split_type: How the layer is split ('column' or 'row')
            
        Returns:
            Tuple of (grad_input, grad_weights, grad_biases)
        """
        if split_type == "column":
            return self.backward_column_parallel(input_tensor, grad_output, weights)
        elif split_type == "row":
            return self.backward_row_parallel(input_tensor, grad_output, weights)
        else:
            raise ValueError("split_type must be 'column' or 'row'")
            
    def create_tensor_parallel_layers(self, layer_sizes):
        """
        Create a set of tensor-parallel linear layers.
        
        Args:
            layer_sizes: List of (input_size, output_size) tuples for each layer
            
        Returns:
            Dictionary of layer parameters
        """
        layers = {}
        for i, (in_size, out_size) in enumerate(layer_sizes):
            # Choose splitting strategy based on layer position
            # Alternate between column and row parallelism for efficient communication
            split_type = "column" if i % 2 == 0 else "row"
            
            # Create full weights and biases
            weights = np.random.randn(in_size, out_size) / np.sqrt(in_size)
            biases = np.zeros(out_size)
            
            # Split weights and biases
            local_weights, local_biases = self.split_linear_layer(weights, biases, split_type)
            
            layers[f"layer_{i}"] = {
                "weights": local_weights,
                "biases": local_biases,
                "split_type": split_type
            }
            
        return layers 