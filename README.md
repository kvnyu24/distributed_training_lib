# Distributed Training Library

A Python library for distributed training that supports both naive and Megatron-style model parallelism, as well as data parallelism.

## Features

- Model Parallelism
  - Naive model parallelism implementation
  - Megatron-style model parallelism implementation
- Data Parallelism support
- Distributed gradient averaging and collection
- Checkpoint management for saving and loading distributed models
- Performance monitoring with communication/computation tracking
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

### Basic Configuration and Training

```python
from distributed_training_lib import ParallelTrainer
from distributed_training_lib.parallel import ModelParallelConfig

# Initialize MPI
config = ModelParallelConfig(
    mp_size=4,  # Model parallel size
    dp_size=2,  # Data parallel size
    is_megatron=True,  # Use Megatron-style parallelism
    in_dim=256,  # Input dimension
    out_dim=512  # Output dimension
)

trainer = ParallelTrainer(config)

# Forward and backward passes
output = trainer.forward(input_data)
grad = trainer.backward(output_grad)
```

### Gradient Management

```python
from distributed_training_lib import GradientManager

# Average gradients across data parallel processes
averaged_gradients = GradientManager.average_gradients(local_gradients, trainer.dp_comm)

# Collect gradients from all processes
all_gradients = GradientManager.collect_gradients(local_gradients, trainer.dp_comm)

# Reduce multiple gradients with or without averaging
gradients_dict = {
    "layer1": layer1_gradients,
    "layer2": layer2_gradients
}
reduced_gradients = GradientManager.reduce_gradients(gradients_dict, trainer.dp_comm, average=True)
```

### Checkpoint Management

```python
from distributed_training_lib import CheckpointManager
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Create checkpoint manager
checkpoint_manager = CheckpointManager(
    checkpoint_dir="checkpoints",
    comm=comm,
    rank=rank
)

# Save model state
state_dict = {
    "weights": np.array(...),
    "bias": np.array(...),
    "learning_rate": 0.01
}
checkpoint_manager.save_checkpoint(state_dict, step=1000)

# Load model state
loaded_state = checkpoint_manager.load_checkpoint()  # Latest checkpoint
loaded_state = checkpoint_manager.load_checkpoint(step=500)  # Specific checkpoint
```

### Performance Monitoring

```python
from distributed_training_lib import PerformanceMonitor
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Create performance monitor
monitor = PerformanceMonitor(
    comm=comm,
    rank=rank,
    log_dir="logs"
)

# Training loop
for step in range(num_steps):
    monitor.step()
    
    # Time computation
    monitor.start_timer("compute")
    # ... computation code ...
    monitor.stop_timer("compute")
    
    # Time communication
    monitor.start_timer("comm")
    # ... communication code ...
    monitor.stop_timer("comm")
    
    # Add custom metrics
    monitor.add_metric("loss", loss_value)
    monitor.increment_counter("num_samples", batch_size)
    
    # Periodically log metrics
    if step % 100 == 0:
        monitor.log_metrics(step)
```

## Documentation

The library is organized into several modules:

- `core/`: Core functionality and base classes
- `parallel/`: Implementation of different parallelism strategies
- `utils/`: Utility functions and helpers
  - `gradient_utils.py`: Utilities for gradient operations
  - `checkpoint_utils.py`: Checkpoint saving and loading
  - `performance_utils.py`: Performance monitoring

For detailed API documentation, please refer to the docstrings in each module.

## Tests

Run the tests using MPI:

```bash
mpirun -n 2 python tests/test_config.py
mpirun -n 2 python tests/test_naive_parallel.py
mpirun -n 2 python tests/test_megatron_parallel.py
mpirun -n 2 python tests/test_data_parallel.py
mpirun -n 2 python tests/test_gradient_utils.py
mpirun -n 2 python tests/test_checkpoint_utils.py
mpirun -n 2 python tests/test_performance_utils.py
```

## License

MIT License
