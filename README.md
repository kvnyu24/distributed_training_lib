# Distributed Training Library

A Python library for distributed training that supports both naive and Megatron-style model parallelism, as well as data parallelism.

## Features

- Model Parallelism
  - Naive model parallelism implementation
  - Megatron-style model parallelism implementation
- Data Parallelism support
- MPI-based communication
- Efficient parallel forward and backward operations

## Installation

```bash
pip install .
```

## Requirements

- Python >= 3.8
- numpy >= 1.26.4
- mpi4py >= 3.1.5
- h5py >= 3.10.0

## Usage

```python
from distributed_training_lib import ParallelTrainer
from distributed_training_lib.parallel import ModelParallelConfig

# Initialize MPI
config = ModelParallelConfig(
    mp_size=4,  # Model parallel size
    dp_size=2,  # Data parallel size
    is_megatron=True  # Use Megatron-style parallelism
)

trainer = ParallelTrainer(config)
# Use the trainer for distributed training
```

## Documentation

The library is organized into several modules:

- `core/`: Core functionality and base classes
- `parallel/`: Implementation of different parallelism strategies
- `utils/`: Utility functions and helpers

For detailed API documentation, please refer to the docstrings in each module.

## License

MIT License
