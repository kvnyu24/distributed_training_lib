"""
Distributed Training Library
===========================

A library for distributed training with model and data parallelism support.
"""

from .core.config import ModelParallelConfig
from .core.trainer import ParallelTrainer
from .parallel.parallel_ops import ParallelOperations

__version__ = "0.1.0"
__all__ = ["ModelParallelConfig", "ParallelTrainer", "ParallelOperations"]
