# Distributed Training Library

A Python library for distributed deep learning training using MPI, supporting various parallelism strategies and optimization techniques.

## Features

- **Model Parallelism**: Split models across multiple devices
  - Naive partitioning (layer-wise)
  - Megatron-style parallelism (attention/MLP blocks)
  - Tensor Parallelism (intra-layer parallelism)
- **Data Parallelism**: Train on different data batches across multiple devices
- **Pipeline Parallelism**: Split models into pipeline stages
- **Memory Optimization**:
  - ZeRO (Zero Redundancy Optimizer) - Memory-efficient data parallelism
  - Activation Checkpointing - Trade computation for memory
- **Utilities**:
  - Distributed gradient averaging and collection
  - Checkpoint management for saving and loading distributed models
  - Performance monitoring with communication/computation tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/distributed_training_lib.git
cd distributed_training_lib

# Install dependencies
pip install -r requirements.txt

# Install the library
pip install -e .
```

## Requirements

- Python 3.6+
- NumPy
- mpi4py
- (Optional) PyTorch or TensorFlow for integration with these frameworks

## Usage

### Basic Configuration and Training

```python
from distributed_training_lib import ParallelTrainer, ModelParallelConfig

# Configure parallelism
config = ModelParallelConfig(
    model_parallel_size=2,  # Split model across 2 devices
    data_parallel_size=2,   # Use 2 data parallel groups
    pipeline_parallel_size=1  # No pipeline parallelism
)

# Create a trainer
trainer = ParallelTrainer(config)

# Set up model (could be a PyTorch or TensorFlow model, or custom implementation)
model = MyModel()

# Train with parallel strategy
trainer.train(model, dataset, epochs=10)
```

### Gradient Management

```python
from distributed_training_lib import GradientManager
from mpi4py import MPI

# Create a gradient manager
gradient_mgr = GradientManager(MPI.COMM_WORLD)

# During training
local_grads = compute_gradients(model, batch)
averaged_grads = gradient_mgr.average_gradients(local_grads)
apply_gradients(model, averaged_grads)
```

### Checkpoint Management

```python
from distributed_training_lib import CheckpointManager
from mpi4py import MPI

# Create a checkpoint manager
checkpoint_mgr = CheckpointManager(MPI.COMM_WORLD, "/path/to/checkpoints")

# Save a checkpoint
model_state = extract_model_state(model)
checkpoint_mgr.save_checkpoint(model_state, step=1000)

# Load a checkpoint
loaded_state = checkpoint_mgr.load_checkpoint(step=1000)
restore_model_state(model, loaded_state)
```

### Performance Monitoring

```python
from distributed_training_lib import PerformanceMonitor
from mpi4py import MPI

# Create a performance monitor
perf_monitor = PerformanceMonitor(MPI.COMM_WORLD)

# During training
perf_monitor.start_timer("forward_pass")
outputs = model.forward(inputs)
perf_monitor.stop_timer("forward_pass")

perf_monitor.start_timer("backward_pass")
gradients = compute_gradients(outputs, labels)
perf_monitor.stop_timer("backward_pass")

# Get performance metrics
metrics = perf_monitor.get_performance_metrics()
print(f"Forward pass time: {metrics['forward_pass']} seconds")
```

### Memory Optimization with ZeRO

```python
from distributed_training_lib import ZeROOptimizer, ZeROStage
from mpi4py import MPI

# Create a ZeRO optimizer (Stage 1, 2, or 3)
optimizer = ZeROOptimizer(
    dp_comm=MPI.COMM_WORLD,
    stage=ZeROStage.STAGE_2,  # Partition optimizer states and gradients
    learning_rate=0.001
)

# Register model parameters
optimizer.register_parameters(model_params)

# Training loop
for epoch in range(epochs):
    # Forward pass
    outputs = model.forward(inputs)
    loss = compute_loss(outputs, labels)
    
    # Compute gradients
    gradients = compute_gradients(loss, model_params)
    
    # Reduce and partition gradients according to ZeRO stage
    reduced_grads = optimizer.reduce_gradients(gradients)
    
    # Update parameters
    updated_params = optimizer.step(model_params, reduced_grads)
    
    # Update model with new parameters
    update_model(model, updated_params)
```

### Tensor Parallelism

```python
from distributed_training_lib import TensorParallelism
from mpi4py import MPI

# Create tensor parallelism handler
tp = TensorParallelism(MPI.COMM_WORLD)

# Split a linear layer across devices
weights = create_large_weight_matrix(1024, 4096)
biases = create_bias_vector(4096)

# Split weights and biases column-wise
local_weights, local_biases = tp.split_linear_layer(weights, biases, split_type="column")

# Forward pass with column-parallel linear layer
output = tp.parallel_linear_forward(input_tensor, local_weights, local_biases, split_type="column")

# Backward pass
grad_input, grad_weights, grad_biases = tp.parallel_linear_backward(
    input_tensor, grad_output, local_weights, split_type="column"
)
```

### Activation Checkpointing

```python
from distributed_training_lib import ActivationCheckpoint

# Create activation checkpointing manager
act_checkpoint = ActivationCheckpoint(checkpoint_segments=2)

# Register layer functions
act_checkpoint.register_function("layer1", layer1_forward, layer1_backward)
act_checkpoint.register_function("layer2", layer2_forward, layer2_backward)

# Forward pass with checkpointing
activations = [input_data]
for i in range(num_layers):
    layer_inputs = [activations[-1], weights[i], biases[i]]
    layer_output = act_checkpoint.checkpoint_activations(
        f"layer_{i}", i, num_layers, layer_inputs
    )
    activations.append(layer_output)

# Backward pass with recomputation
grad_output = compute_loss_gradient(activations[-1], targets)
for i in range(num_layers-1, -1, -1):
    layer_inputs = [activations[i], weights[i], biases[i]]
    grad_inputs = act_checkpoint.backward_pass(
        f"layer_{i}", i, num_layers, grad_output
    )
    grad_output = grad_inputs[0]
```

## Tests

Run the tests using MPI:

```bash
# Run all tests
./run_tests.sh

# Run specific tests
mpirun -n 2 python tests/test_config.py
mpirun -n 2 python tests/test_naive_parallel.py
mpirun -n 2 python tests/test_megatron_parallel.py
mpirun -n 2 python tests/test_data_parallel.py
mpirun -n 2 python tests/test_pipeline_parallel.py
mpirun -n 2 python tests/test_gradient_utils.py
mpirun -n 2 python tests/test_checkpoint_utils.py
mpirun -n 2 python tests/test_performance_utils.py
mpirun -n 2 python tests/test_zero_optimizer.py
mpirun -n 2 python tests/test_tensor_parallel.py
mpirun -n 2 python tests/test_activation_checkpoint.py
```

## Documentation

For detailed documentation, see the `docs/` directory or the docstrings in the code.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
