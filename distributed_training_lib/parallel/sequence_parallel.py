import numpy as np
from mpi4py import MPI
import math
import logging

logger = logging.getLogger(__name__)

class SequenceParallelism:
    """
    Implementation of sequence parallelism for distributed training.
    
    Sequence parallelism splits the sequence dimension (e.g., the token dimension in NLP models)
    across multiple devices, allowing for efficient processing of longer sequences.
    This is particularly useful for transformer-based models with long contexts.
    
    The key difference from tensor parallelism is that sequence parallelism partitions
    along the sequence (batch) dimension rather than model parameters.
    
    References:
    - Reducing Activation Recomputation in Large Transformer Models
      (https://arxiv.org/abs/2205.05198)
    """
    
    def __init__(self, comm, seq_dim=1):
        """
        Initialize sequence parallelism.
        
        Args:
            comm: MPI communicator for sequence parallelism
            seq_dim: Dimension of the sequence to split (default is 1, assuming batch_size x seq_len x ...)
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.world_size = comm.Get_size()
        self.seq_dim = seq_dim
        
        logger.info(f"Initialized SequenceParallelism with {self.world_size} processes")
    
    def split_sequence(self, tensor):
        """
        Split a tensor along the sequence dimension.
        
        Args:
            tensor: NumPy array with shape (batch_size, seq_len, ...)
            
        Returns:
            Local portion of the tensor for this process
        """
        if tensor.ndim <= self.seq_dim:
            raise ValueError(f"Tensor has {tensor.ndim} dimensions, but seq_dim is {self.seq_dim}")
        
        # Get the sequence length
        seq_len = tensor.shape[self.seq_dim]
        
        # Calculate local sequence length for this process
        local_seq_len = seq_len // self.world_size
        remainder = seq_len % self.world_size
        
        # Adjust for remainder
        if self.rank < remainder:
            local_seq_len += 1
            start_idx = self.rank * local_seq_len
        else:
            start_idx = self.rank * local_seq_len + remainder
        
        # Create slice indices
        slice_indices = [slice(None)] * tensor.ndim
        slice_indices[self.seq_dim] = slice(start_idx, start_idx + local_seq_len)
        
        # Get local portion
        local_tensor = tensor[tuple(slice_indices)].copy()
        
        logger.debug(f"Rank {self.rank}: Split sequence from length {seq_len} to {local_tensor.shape[self.seq_dim]}")
        
        return local_tensor
    
    def gather_sequence(self, local_tensor):
        """
        Gather a tensor that was previously split along the sequence dimension.
        
        Args:
            local_tensor: Local tensor portion from this process
            
        Returns:
            Full gathered tensor
        """
        # Get local sequence length
        local_seq_len = local_tensor.shape[self.seq_dim]
        
        # Gather local sequence lengths from all processes
        local_seq_len_arr = np.array([local_seq_len], dtype=np.int32)
        all_seq_lens = np.zeros(self.world_size, dtype=np.int32)
        
        self.comm.Allgather(local_seq_len_arr, all_seq_lens)
        
        # Calculate displacements for gathering
        displacements = np.zeros(self.world_size, dtype=np.int32)
        for i in range(1, self.world_size):
            displacements[i] = displacements[i-1] + all_seq_lens[i-1]
        
        # Calculate full tensor shape and create buffer
        full_shape = list(local_tensor.shape)
        full_shape[self.seq_dim] = np.sum(all_seq_lens)
        full_tensor = np.zeros(full_shape, dtype=local_tensor.dtype)
        
        # Create slice for this process's portion in the full tensor
        slice_indices = [slice(None)] * local_tensor.ndim
        slice_indices[self.seq_dim] = slice(displacements[self.rank], 
                                         displacements[self.rank] + local_seq_len)
        
        # Place local tensor in the full tensor
        full_tensor[tuple(slice_indices)] = local_tensor
        
        # Gather from all processes
        self.comm.Allreduce(MPI.IN_PLACE, full_tensor, op=MPI.SUM)
        
        logger.debug(f"Rank {self.rank}: Gathered sequence to length {full_tensor.shape[self.seq_dim]}")
        
        return full_tensor
    
    def sequence_parallel_attention(self, query, key, value, mask=None):
        """
        Sequence-parallel attention computation.
        
        Each process computes attention for a portion of the sequence.
        
        Args:
            query: Query tensor (batch_size, seq_len, hidden_dim)
            key: Key tensor (batch_size, seq_len, hidden_dim)
            value: Value tensor (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask
            
        Returns:
            Attention output tensor
        """
        # Split query, key, value along sequence dimension
        local_query = self.split_sequence(query)
        local_key = key  # Full key needed for attention
        local_value = value  # Full value needed for attention
        
        # Split mask if provided
        local_mask = None
        if mask is not None:
            local_mask = self.split_sequence(mask)
        
        # Compute local attention scores: (batch_size, local_seq_len, seq_len)
        scale = 1.0 / np.sqrt(local_query.shape[-1])
        local_scores = np.matmul(local_query, local_key.transpose(0, 2, 1)) * scale
        
        # Apply mask if needed
        if local_mask is not None:
            local_scores = local_scores + local_mask
        
        # Apply softmax along the last dimension
        local_attention_weights = self._softmax(local_scores, axis=-1)
        
        # Compute local weighted sum
        local_output = np.matmul(local_attention_weights, local_value)
        
        # Gather outputs from all processes
        output = self.gather_sequence(local_output)
        
        return output
    
    def sequence_parallel_mlp(self, input_tensor, weights1, biases1, weights2, biases2):
        """
        Sequence-parallel MLP computation.
        
        Each process computes MLP for a portion of the sequence.
        
        Args:
            input_tensor: Input tensor (batch_size, seq_len, hidden_dim)
            weights1: First layer weights (hidden_dim, intermediate_dim)
            biases1: First layer biases (intermediate_dim)
            weights2: Second layer weights (intermediate_dim, hidden_dim)
            biases2: Second layer biases (hidden_dim)
            
        Returns:
            MLP output tensor
        """
        # Split input along sequence dimension
        local_input = self.split_sequence(input_tensor)
        
        # First linear layer
        # (batch_size, local_seq_len, hidden_dim) x (hidden_dim, intermediate_dim) -> (batch_size, local_seq_len, intermediate_dim)
        local_intermediate = np.matmul(local_input, weights1) + biases1
        
        # Apply GELU activation
        local_intermediate = self._gelu(local_intermediate)
        
        # Second linear layer
        # (batch_size, local_seq_len, intermediate_dim) x (intermediate_dim, hidden_dim) -> (batch_size, local_seq_len, hidden_dim)
        local_output = np.matmul(local_intermediate, weights2) + biases2
        
        # Gather outputs from all processes
        output = self.gather_sequence(local_output)
        
        return output
    
    def sequence_parallel_layer_norm(self, input_tensor, gamma, beta, epsilon=1e-12):
        """
        Sequence-parallel layer normalization.
        
        Each process computes layer norm for a portion of the sequence.
        
        Args:
            input_tensor: Input tensor (batch_size, seq_len, hidden_dim)
            gamma: Scale parameter (hidden_dim)
            beta: Shift parameter (hidden_dim)
            epsilon: Small constant for numerical stability
            
        Returns:
            Layer-normalized output tensor
        """
        # Split input along sequence dimension
        local_input = self.split_sequence(input_tensor)
        
        # Calculate mean and variance along the last dimension
        mean = np.mean(local_input, axis=-1, keepdims=True)
        variance = np.var(local_input, axis=-1, keepdims=True)
        
        # Normalize
        local_output = (local_input - mean) / np.sqrt(variance + epsilon)
        
        # Scale and shift
        local_output = local_output * gamma + beta
        
        # Gather outputs from all processes
        output = self.gather_sequence(local_output)
        
        return output
    
    def sequence_parallel_transformer_layer(self, input_tensor, attention_params, mlp_params, 
                                           layer_norm_params, mask=None):
        """
        Sequence-parallel transformer layer computation.
        
        Args:
            input_tensor: Input tensor (batch_size, seq_len, hidden_dim)
            attention_params: Dict with Q, K, V, and output projection weights and biases
            mlp_params: Dict with MLP weights and biases
            layer_norm_params: Dict with layer norm parameters
            mask: Optional attention mask
            
        Returns:
            Transformer layer output tensor
        """
        # Layer norm 1
        ln1_output = self.sequence_parallel_layer_norm(
            input_tensor, 
            layer_norm_params['ln1_gamma'], 
            layer_norm_params['ln1_beta']
        )
        
        # Self-attention
        # Project to Q, K, V
        local_ln1 = self.split_sequence(ln1_output)
        local_query = np.matmul(local_ln1, attention_params['wq']) + attention_params['bq']
        local_key = np.matmul(local_ln1, attention_params['wk']) + attention_params['bk']
        local_value = np.matmul(local_ln1, attention_params['wv']) + attention_params['bv']
        
        # Gather for attention computation (needed for all processes)
        query = self.gather_sequence(local_query)
        key = self.gather_sequence(local_key)
        value = self.gather_sequence(local_value)
        
        # Compute attention
        attention_output = self.sequence_parallel_attention(query, key, value, mask)
        
        # Project output
        local_attention_output = self.split_sequence(attention_output)
        local_projected = np.matmul(local_attention_output, attention_params['wo']) + attention_params['bo']
        projected_output = self.gather_sequence(local_projected)
        
        # Residual connection
        res1 = input_tensor + projected_output
        
        # Layer norm 2
        ln2_output = self.sequence_parallel_layer_norm(
            res1, 
            layer_norm_params['ln2_gamma'], 
            layer_norm_params['ln2_beta']
        )
        
        # MLP
        mlp_output = self.sequence_parallel_mlp(
            ln2_output,
            mlp_params['w1'], mlp_params['b1'],
            mlp_params['w2'], mlp_params['b2']
        )
        
        # Residual connection
        output = res1 + mlp_output
        
        return output
    
    def _softmax(self, x, axis=-1):
        """Simple softmax implementation."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _gelu(self, x):
        """Simple GELU activation implementation."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        
    def create_sequence_parallel_transformer(self, config):
        """
        Create a sequence-parallel transformer with random weights.
        
        Args:
            config: Dict with model configuration (hidden_dim, intermediate_dim, num_layers, etc.)
            
        Returns:
            Dict with transformer parameters
        """
        hidden_dim = config.get('hidden_dim', 768)
        intermediate_dim = config.get('intermediate_dim', 3072)
        num_layers = config.get('num_layers', 12)
        
        # Initialize random parameters for demonstration
        np.random.seed(42)
        
        model_params = {
            'layers': []
        }
        
        for _ in range(num_layers):
            layer_params = {
                'attention': {
                    'wq': np.random.randn(hidden_dim, hidden_dim) * 0.02,
                    'wk': np.random.randn(hidden_dim, hidden_dim) * 0.02,
                    'wv': np.random.randn(hidden_dim, hidden_dim) * 0.02,
                    'wo': np.random.randn(hidden_dim, hidden_dim) * 0.02,
                    'bq': np.zeros(hidden_dim),
                    'bk': np.zeros(hidden_dim),
                    'bv': np.zeros(hidden_dim),
                    'bo': np.zeros(hidden_dim)
                },
                'mlp': {
                    'w1': np.random.randn(hidden_dim, intermediate_dim) * 0.02,
                    'w2': np.random.randn(intermediate_dim, hidden_dim) * 0.02,
                    'b1': np.zeros(intermediate_dim),
                    'b2': np.zeros(hidden_dim)
                },
                'layer_norm': {
                    'ln1_gamma': np.ones(hidden_dim),
                    'ln1_beta': np.zeros(hidden_dim),
                    'ln2_gamma': np.ones(hidden_dim),
                    'ln2_beta': np.zeros(hidden_dim)
                }
            }
            model_params['layers'].append(layer_params)
        
        model_params['embeddings'] = np.random.randn(config.get('vocab_size', 30000), hidden_dim) * 0.02
        model_params['embedding_bias'] = np.zeros(hidden_dim)
        model_params['output_bias'] = np.zeros(config.get('vocab_size', 30000))
        
        return model_params
    
    def sequence_parallel_forward(self, input_ids, model_params, attention_mask=None):
        """
        Forward pass for sequence-parallel transformer.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            model_params: Dict with transformer parameters
            attention_mask: Optional attention mask
            
        Returns:
            Model output logits
        """
        # Get embeddings
        batch_size, seq_len = input_ids.shape
        hidden_dim = model_params['embeddings'].shape[1]
        
        # Convert input_ids to one-hot
        embeddings = model_params['embeddings'][input_ids.reshape(-1)].reshape(batch_size, seq_len, hidden_dim)
        hidden_states = embeddings + model_params['embedding_bias']
        
        # Process through transformer layers
        for layer_params in model_params['layers']:
            hidden_states = self.sequence_parallel_transformer_layer(
                hidden_states,
                layer_params['attention'],
                layer_params['mlp'],
                layer_params['layer_norm'],
                attention_mask
            )
        
        # Project to logits
        local_hidden = self.split_sequence(hidden_states)
        local_logits = np.matmul(local_hidden, model_params['embeddings'].transpose())
        local_logits = local_logits + model_params['output_bias']
        
        # Gather logits
        logits = self.gather_sequence(local_logits)
        
        return logits 