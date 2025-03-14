import numpy as np
from mpi4py import MPI
from typing import List, Dict, Optional, Tuple

from .expert_parallel import ExpertParallelism

class MixtureOfExperts:
    """
    Mixture of Experts (MoE) model implementation using expert parallelism.
    
    This class implements a simplified MoE model where inputs are routed to
    different experts based on a gating mechanism. The experts are distributed
    across different devices using expert parallelism.
    
    MoE layers consist of:
    1. A router/gate that determines which experts process each input
    2. A set of experts (typically feed-forward networks)
    3. A mechanism to combine expert outputs
    
    This implementation follows the architecture described in papers like:
    - GShard (https://arxiv.org/abs/2006.16668)
    - Switch Transformers (https://arxiv.org/abs/2101.03961)
    - DeepSeek-V3 (https://github.com/deepseek-ai/DeepSeek-V3)
    """
    
    def __init__(self, comm, num_experts, hidden_size, ffn_hidden_size, 
                 num_experts_per_token=2, activation_fn=None):
        """
        Initialize the MoE model.
        
        Args:
            comm: MPI communicator
            num_experts: Total number of experts
            hidden_size: Size of input/output embeddings
            ffn_hidden_size: Size of expert hidden layers
            num_experts_per_token: Number of experts to route each token to
            activation_fn: Activation function for experts
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.world_size = comm.Get_size()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_experts_per_token = num_experts_per_token
        
        # Initialize expert parallelism
        self.ep = ExpertParallelism(comm, num_experts)
        
        # Initialize router parameters
        # Simple linear layer that produces logits for expert selection
        self.router_weights = np.random.randn(hidden_size, num_experts) * 0.02
        
        # Create expert parameters
        self.moe_layer = self.ep.create_moe_layer(
            hidden_size, ffn_hidden_size, activation_fn
        )
        
        if self.rank == 0:
            print(f"Initialized MoE model with {num_experts} experts")
            print(f"Hidden size: {hidden_size}, FFN hidden size: {ffn_hidden_size}")
            print(f"Experts per token: {num_experts_per_token}")
    
    def forward(self, input_tensor):
        """
        Forward pass through the MoE model.
        
        Args:
            input_tensor: Input tensor of shape [batch_size, hidden_size]
            
        Returns:
            output_tensor: Output tensor of shape [batch_size, hidden_size]
        """
        batch_size, hidden_dim = input_tensor.shape
        
        # 1. Route tokens to experts (gating)
        router_logits = np.matmul(input_tensor, self.router_weights)
        expert_indices, expert_weights = self.ep.top_k_gating(
            router_logits, k=self.num_experts_per_token
        )
        
        # 2. Process tokens with experts
        output_tensor = self.ep.forward(
            input_tensor, expert_indices, expert_weights, self.moe_layer
        )
        
        return output_tensor, (expert_indices, expert_weights)
    
    def compute_load_balancing_loss(self, expert_indices, expert_weights):
        """
        Compute the load balancing loss to ensure experts are utilized evenly.
        
        Args:
            expert_indices: Indices of selected experts [batch_size, k]
            expert_weights: Weights for selected experts [batch_size, k]
            
        Returns:
            load_balance_loss: A scalar loss term to encourage balanced expert usage
        """
        batch_size = expert_indices.shape[0]
        
        # Calculate the fraction of tokens routed to each expert
        expert_usage = np.zeros(self.num_experts)
        for i in range(batch_size):
            for j in range(expert_indices.shape[1]):
                expert_idx = expert_indices[i, j]
                weight = expert_weights[i, j]
                expert_usage[expert_idx] += weight
        
        # Normalize by batch size
        expert_usage /= batch_size
        
        # Ideal usage: each expert should process (num_experts_per_token / num_experts) of tokens
        ideal_usage = self.num_experts_per_token / self.num_experts
        
        # Compute load balancing loss: squared difference from ideal
        load_balance_loss = np.mean((expert_usage - ideal_usage) ** 2)
        
        # Gather expert usage stats from all ranks
        all_expert_usage = np.zeros(self.num_experts)
        self.comm.Allreduce(expert_usage, all_expert_usage, op=MPI.SUM)
        all_expert_usage /= self.world_size
        
        if self.rank == 0:
            print(f"Expert usage: {all_expert_usage}")
            print(f"Load balancing loss: {load_balance_loss}")
        
        return load_balance_loss
    
    def update_parameters(self, input_tensor, output_tensor, grad_output, learning_rate=0.01):
        """
        Simple parameter update using gradient descent.
        
        Args:
            input_tensor: Input tensor from forward pass
            output_tensor: Output tensor from forward pass
            grad_output: Gradient of loss with respect to output
            learning_rate: Learning rate for gradient descent
            
        Returns:
            None (updates parameters in-place)
        """
        batch_size = input_tensor.shape[0]
        
        # 1. Update router parameters
        # This is a simplified gradient computation for the router
        # In practice, you'd compute this more carefully
        router_grad = np.zeros_like(self.router_weights)
        self.router_weights -= learning_rate * router_grad
        
        # 2. For each local expert, update its parameters
        # This is also simplified - in practice you'd compute proper gradients
        for expert_idx, expert_params in self.moe_layer.items():
            # Update first layer weights
            w1_grad = np.zeros_like(expert_params["w1"])
            b1_grad = np.zeros_like(expert_params["b1"])
            expert_params["w1"] -= learning_rate * w1_grad
            expert_params["b1"] -= learning_rate * b1_grad
            
            # Update second layer weights
            w2_grad = np.zeros_like(expert_params["w2"])
            b2_grad = np.zeros_like(expert_params["b2"])
            expert_params["w2"] -= learning_rate * w2_grad
            expert_params["b2"] -= learning_rate * b2_grad 