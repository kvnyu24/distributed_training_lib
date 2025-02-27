from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelParallelConfig:
    """Configuration class for model parallel training.

    Attributes:
        mp_size: Model parallel size
        dp_size: Data parallel size
        is_megatron: Whether to use Megatron-style model parallelism
        in_dim: Input dimension for the model
        out_dim: Output dimension for the model
    """
    mp_size: int
    dp_size: int
    is_megatron: bool = False
    in_dim: Optional[int] = None
    out_dim: Optional[int] = None

    def __post_init__(self):
        if self.mp_size <= 0:
            raise ValueError("Model parallel size must be positive")
        if self.dp_size <= 0:
            raise ValueError("Data parallel size must be positive") 