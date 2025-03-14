class ParallelConfig:
    """
    Configuration class for parallel training settings.
    
    This class provides a centralized way to configure various parallelism strategies
    including model parallelism, data parallelism, pipeline parallelism, tensor parallelism,
    and expert parallelism.
    """
    
    def __init__(self, 
                 model_parallel_size=1, 
                 data_parallel_size=1,
                 pipeline_parallel_size=1,
                 tensor_parallel_size=1,
                 expert_parallel_size=1,
                 num_experts=0,
                 experts_per_token=0):
        """
        Initialize the parallel configuration.
        
        Args:
            model_parallel_size: Number of model parallel partitions
            data_parallel_size: Number of data parallel partitions
            pipeline_parallel_size: Number of pipeline parallel stages
            tensor_parallel_size: Number of tensor parallel partitions
            expert_parallel_size: Number of expert parallel partitions
            num_experts: Total number of experts in MoE models
            experts_per_token: Number of experts to route each token to in MoE models
        """
        self.model_parallel_size = model_parallel_size
        self.data_parallel_size = data_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.expert_parallel_size = expert_parallel_size
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        
    def validate(self, world_size):
        """
        Validate that the configuration is compatible with the given world size.
        
        Args:
            world_size: Total number of processes
            
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Check that the product of parallel sizes equals world size
        expected_size = (self.model_parallel_size * 
                         self.data_parallel_size * 
                         self.pipeline_parallel_size * 
                         self.tensor_parallel_size)
        
        if self.expert_parallel_size > 1:
            expected_size *= self.expert_parallel_size
        
        return expected_size == world_size
    
    def __str__(self):
        """Return a string representation of the configuration."""
        return (f"ParallelConfig(model_parallel_size={self.model_parallel_size}, "
                f"data_parallel_size={self.data_parallel_size}, "
                f"pipeline_parallel_size={self.pipeline_parallel_size}, "
                f"tensor_parallel_size={self.tensor_parallel_size}, "
                f"expert_parallel_size={self.expert_parallel_size}, "
                f"num_experts={self.num_experts}, "
                f"experts_per_token={self.experts_per_token})") 