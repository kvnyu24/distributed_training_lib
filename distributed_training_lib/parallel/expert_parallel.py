import numpy as np
from mpi4py import MPI
import math
from typing import Tuple, List, Dict, Optional, Union

class ExpertParallelism:
    """
    Implementation of expert parallelism for Mixture of Experts (MoE) models.
    
    Expert parallelism distributes different "expert" networks across multiple devices,
    allowing for efficient training of large sparse models. This is particularly useful 
    for MoE models where only a subset of experts process each input token.
    
    Each device hosts a subset of experts, and tokens are routed to the appropriate
    expert based on a routing algorithm (typically top-k gating). This implementation 
    provides functions for:
    
    1. Expert distribution: Assigning experts to devices
    2. Dispatch: Routing tokens to the correct experts
    3. Combine: Collecting results from experts
    
    References:
    - GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding
      (https://arxiv.org/abs/2006.16668)
    - DeepSeek-V3 (https://github.com/deepseek-ai/DeepSeek-V3)
    - DeepEP (https://github.com/deepseek-ai/DeepEP)
    """
    
    def __init__(self, comm, num_experts, num_experts_per_rank=None):
        """
        Initialize expert parallelism.
        
        Args:
            comm: MPI communicator for expert parallelism
            num_experts: Total number of experts in the model
            num_experts_per_rank: Number of experts per rank (default: evenly distributed)
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.world_size = comm.Get_size()
        self.num_experts = num_experts
        
        # Determine number of experts per rank
        if num_experts_per_rank is None:
            self.num_experts_per_rank = num_experts // self.world_size
            if self.rank < (num_experts % self.world_size):
                self.num_experts_per_rank += 1
        else:
            self.num_experts_per_rank = num_experts_per_rank
        
        # Map experts to ranks
        self.expert_to_rank = {}
        self.local_expert_indices = []
        
        # Assign experts to ranks
        experts_assigned = 0
        for r in range(self.world_size):
            experts_on_rank = num_experts // self.world_size
            if r < (num_experts % self.world_size):
                experts_on_rank += 1
                
            for i in range(experts_assigned, experts_assigned + experts_on_rank):
                self.expert_to_rank[i] = r
                if r == self.rank:
                    self.local_expert_indices.append(i)
            
            experts_assigned += experts_on_rank
        
        print(f"Rank {self.rank}: Assigned {len(self.local_expert_indices)} experts: {self.local_expert_indices}")
    
    def get_expert_rank(self, expert_idx):
        """Get the rank that hosts a specific expert."""
        return self.expert_to_rank.get(expert_idx, -1)
    
    def dispatch(self, input_tensor, expert_indices, expert_weights=None):
        """
        Dispatch tokens to their corresponding experts across devices.
        
        Args:
            input_tensor: Input tensor of shape [batch_size, hidden_dim]
            expert_indices: Tensor of shape [batch_size, k] indicating which experts to route each token to
            expert_weights: Optional tensor of shape [batch_size, k] with weights for each expert
            
        Returns:
            local_input_tensor: Input tensor for local experts
            local_expert_indices: Indices of local experts
            local_token_indices: Indices of tokens assigned to local experts
            local_expert_weights: Weights for local expert computation
        """
        batch_size = input_tensor.shape[0]
        hidden_dim = input_tensor.shape[1]
        
        # Default weights if not provided
        if expert_weights is None:
            k = expert_indices.shape[1]
            expert_weights = np.ones((batch_size, k)) / k
        
        # Count how many tokens to send to each rank
        tokens_per_rank = [0] * self.world_size
        for i in range(batch_size):
            for j in range(expert_indices.shape[1]):
                expert_idx = expert_indices[i, j]
                rank = self.get_expert_rank(expert_idx)
                if rank >= 0:
                    tokens_per_rank[rank] += 1
        
        # Gather token counts from all ranks
        all_token_counts = self.comm.allgather(tokens_per_rank)
        
        # Determine which tokens and data to send to each rank
        send_data = [[] for _ in range(self.world_size)]
        send_token_indices = [[] for _ in range(self.world_size)]
        send_expert_indices = [[] for _ in range(self.world_size)]
        send_expert_weights = [[] for _ in range(self.world_size)]
        
        for i in range(batch_size):
            for j in range(expert_indices.shape[1]):
                expert_idx = expert_indices[i, j]
                rank = self.get_expert_rank(expert_idx)
                if rank >= 0:
                    send_data[rank].append(input_tensor[i])
                    send_token_indices[rank].append(i)
                    send_expert_indices[rank].append(expert_idx)
                    send_expert_weights[rank].append(expert_weights[i, j])
        
        # Convert lists to arrays for sending
        send_data_arrays = []
        for r in range(self.world_size):
            if send_data[r]:
                send_data_arrays.append(np.stack(send_data[r]))
            else:
                send_data_arrays.append(np.zeros((0, hidden_dim)))
        
        # Exchange data
        recv_data_arrays = self.comm.alltoall(send_data_arrays)
        
        # Send token indices
        send_token_indices_arrays = []
        for r in range(self.world_size):
            if send_token_indices[r]:
                send_token_indices_arrays.append(np.array(send_token_indices[r], dtype=np.int32))
            else:
                send_token_indices_arrays.append(np.zeros(0, dtype=np.int32))
        
        recv_token_indices_arrays = self.comm.alltoall(send_token_indices_arrays)
        
        # Send expert indices
        send_expert_indices_arrays = []
        for r in range(self.world_size):
            if send_expert_indices[r]:
                send_expert_indices_arrays.append(np.array(send_expert_indices[r], dtype=np.int32))
            else:
                send_expert_indices_arrays.append(np.zeros(0, dtype=np.int32))
        
        recv_expert_indices_arrays = self.comm.alltoall(send_expert_indices_arrays)
        
        # Send expert weights
        send_expert_weights_arrays = []
        for r in range(self.world_size):
            if send_expert_weights[r]:
                send_expert_weights_arrays.append(np.array(send_expert_weights[r], dtype=np.float32))
            else:
                send_expert_weights_arrays.append(np.zeros(0, dtype=np.float32))
        
        recv_expert_weights_arrays = self.comm.alltoall(send_expert_weights_arrays)
        
        # Concatenate all received data
        local_input_tensor = np.concatenate(recv_data_arrays, axis=0) if any(arr.shape[0] > 0 for arr in recv_data_arrays) else np.zeros((0, hidden_dim))
        local_token_indices = np.concatenate(recv_token_indices_arrays) if any(arr.shape[0] > 0 for arr in recv_token_indices_arrays) else np.zeros(0, dtype=np.int32)
        local_expert_indices = np.concatenate(recv_expert_indices_arrays) if any(arr.shape[0] > 0 for arr in recv_expert_indices_arrays) else np.zeros(0, dtype=np.int32)
        local_expert_weights = np.concatenate(recv_expert_weights_arrays) if any(arr.shape[0] > 0 for arr in recv_expert_weights_arrays) else np.zeros(0, dtype=np.float32)
        
        # Map global expert indices to local indices
        expert_to_local = {expert: i for i, expert in enumerate(self.local_expert_indices)}
        local_expert_indices = np.array([expert_to_local.get(idx, -1) for idx in local_expert_indices], dtype=np.int32)
        
        return local_input_tensor, local_expert_indices, local_token_indices, local_expert_weights
    
    def combine(self, local_output_tensor, local_token_indices, local_expert_weights, batch_size, hidden_dim):
        """
        Combine outputs from experts back to their original positions.
        
        Args:
            local_output_tensor: Output tensor from local experts
            local_token_indices: Indices of tokens assigned to local experts
            local_expert_weights: Weights for each expert output
            batch_size: Original batch size
            hidden_dim: Hidden dimension size
            
        Returns:
            combined_output: Combined output tensor of shape [batch_size, hidden_dim]
        """
        # Weighted local outputs
        weighted_local_output = local_output_tensor * local_expert_weights[:, np.newaxis]
        
        # Prepare data to send back to original ranks
        send_data = [[] for _ in range(self.world_size)]
        send_token_indices = [[] for _ in range(self.world_size)]
        
        # Determine the original rank for each token
        token_ranks = self.comm.allgather([self.rank] * batch_size)
        token_to_rank = {}
        for r in range(self.world_size):
            for i in range(len(token_ranks[r])):
                token_to_rank[i] = r
        
        # Organize data to send back
        for i in range(len(local_token_indices)):
            token_idx = local_token_indices[i]
            rank = token_to_rank.get(token_idx, -1)
            if rank >= 0:
                send_data[rank].append(weighted_local_output[i])
                send_token_indices[rank].append(token_idx)
        
        # Convert lists to arrays for sending
        send_data_arrays = []
        for r in range(self.world_size):
            if send_data[r]:
                send_data_arrays.append(np.stack(send_data[r]))
            else:
                send_data_arrays.append(np.zeros((0, hidden_dim)))
        
        # Exchange data
        recv_data_arrays = self.comm.alltoall(send_data_arrays)
        
        # Send token indices
        send_token_indices_arrays = []
        for r in range(self.world_size):
            if send_token_indices[r]:
                send_token_indices_arrays.append(np.array(send_token_indices[r], dtype=np.int32))
            else:
                send_token_indices_arrays.append(np.zeros(0, dtype=np.int32))
        
        recv_token_indices_arrays = self.comm.alltoall(send_token_indices_arrays)
        
        # Initialize combined output tensor
        combined_output = np.zeros((batch_size, hidden_dim))
        
        # Combine received outputs to their original positions
        for r in range(self.world_size):
            if recv_data_arrays[r].shape[0] > 0:
                for i in range(recv_data_arrays[r].shape[0]):
                    token_idx = recv_token_indices_arrays[r][i]
                    combined_output[token_idx] += recv_data_arrays[r][i]
        
        return combined_output
    
    def create_moe_layer(self, hidden_size, ffn_hidden_size, activation_fn=None):
        """
        Create a Mixture of Experts (MoE) layer with local experts.
        
        Args:
            hidden_size: Input/output dimension size
            ffn_hidden_size: Hidden dimension size for the feedforward network
            activation_fn: Activation function (default: ReLU)
            
        Returns:
            Dictionary of MoE layer parameters for local experts
        """
        if activation_fn is None:
            activation_fn = lambda x: np.maximum(0, x)  # ReLU
        
        experts = {}
        for i, expert_idx in enumerate(self.local_expert_indices):
            # Initialize weights for this expert (in practice, you might want to use a better initialization)
            w1 = np.random.randn(hidden_size, ffn_hidden_size) * 0.02
            b1 = np.zeros(ffn_hidden_size)
            w2 = np.random.randn(ffn_hidden_size, hidden_size) * 0.02
            b2 = np.zeros(hidden_size)
            
            experts[expert_idx] = {
                "w1": w1,
                "b1": b1,
                "w2": w2,
                "b2": b2,
                "activation_fn": activation_fn
            }
            
        return experts
    
    def forward_expert(self, x, expert_params):
        """
        Forward pass through a single expert.
        
        Args:
            x: Input tensor
            expert_params: Expert parameters including weights and activation function
            
        Returns:
            Output tensor after passing through the expert
        """
        # First linear layer
        h = np.matmul(x, expert_params["w1"]) + expert_params["b1"]
        
        # Activation function
        h = expert_params["activation_fn"](h)
        
        # Second linear layer
        output = np.matmul(h, expert_params["w2"]) + expert_params["b2"]
        
        return output
    
    def forward(self, input_tensor, expert_indices, expert_weights, moe_layer_params):
        """
        Forward pass through the MoE layer using expert parallelism.
        
        Args:
            input_tensor: Input tensor of shape [batch_size, hidden_dim]
            expert_indices: Tensor of shape [batch_size, k] indicating which experts to route each token to
            expert_weights: Tensor of shape [batch_size, k] with weights for each expert
            moe_layer_params: Dictionary of MoE layer parameters
            
        Returns:
            output_tensor: Output tensor of shape [batch_size, hidden_dim]
        """
        batch_size, hidden_dim = input_tensor.shape
        
        # Dispatch inputs to their corresponding experts
        local_input, local_expert_indices, local_token_indices, local_weights = self.dispatch(
            input_tensor, expert_indices, expert_weights
        )
        
        # Process inputs with local experts
        local_outputs = np.zeros_like(local_input)
        for i in range(len(local_input)):
            expert_idx = self.local_expert_indices[local_expert_indices[i]]
            expert_params = moe_layer_params.get(expert_idx)
            if expert_params:
                local_outputs[i] = self.forward_expert(local_input[i], expert_params)
        
        # Combine outputs back
        output_tensor = self.combine(
            local_outputs, local_token_indices, local_weights, batch_size, hidden_dim
        )
        
        return output_tensor
    
    def top_k_gating(self, logits, k=2):
        """
        Simple top-k gating mechanism for MoE.
        
        Args:
            logits: Logits tensor of shape [batch_size, num_experts]
            k: Number of experts to select for each token
            
        Returns:
            expert_indices: Indices of selected experts of shape [batch_size, k]
            expert_weights: Weights for selected experts of shape [batch_size, k]
        """
        batch_size = logits.shape[0]
        
        # Apply softmax
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        
        # Select top-k experts
        expert_indices = np.argsort(-probs, axis=1)[:, :k]
        
        # Get corresponding probabilities
        expert_weights = np.zeros((batch_size, k))
        for i in range(batch_size):
            for j in range(k):
                expert_weights[i, j] = probs[i, expert_indices[i, j]]
        
        # Normalize weights
        expert_weights = expert_weights / np.sum(expert_weights, axis=1, keepdims=True)
        
        return expert_indices, expert_weights 