"""
Distributed Training Library
===========================

A library for distributed training with model and data parallelism support.
"""

from .core.config import ModelParallelConfig
from .core.trainer import ParallelTrainer
from .parallel.parallel_ops import ParallelOperations
from .utils.gradient_utils import GradientManager
from .utils.checkpoint_utils import CheckpointManager
from .utils.performance_utils import PerformanceMonitor, Timer, measure_allreduce_bandwidth

__version__ = "0.1.0"
__all__ = [
    "ModelParallelConfig", 
    "ParallelTrainer", 
    "ParallelOperations", 
    "GradientManager",
    "CheckpointManager",
    "PerformanceMonitor",
    "Timer",
    "measure_allreduce_bandwidth"
]
