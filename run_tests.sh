#!/bin/bash

# Set the number of processes to use for tests
NUM_PROCESSES=2

# Run basic configuration tests
echo "Running basic configuration tests..."
mpirun -n $NUM_PROCESSES python tests/test_config.py

# Run naive model parallelism tests
echo "Running naive model parallelism tests..."
mpirun -n $NUM_PROCESSES python tests/test_naive_parallel.py

# Run megatron model parallelism tests
echo "Running megatron model parallelism tests..."
mpirun -n $NUM_PROCESSES python tests/test_megatron_parallel.py

# Run data parallelism tests
echo "Running data parallelism tests..."
mpirun -n $NUM_PROCESSES python tests/test_data_parallel.py

# Run gradient utilities tests
echo "Running gradient utilities tests..."
mpirun -n $NUM_PROCESSES python tests/test_gradient_utils.py

# Run checkpoint utilities tests
echo "Running checkpoint utilities tests..."
mpirun -n $NUM_PROCESSES python tests/test_checkpoint_utils.py

# Run performance utilities tests
echo "Running performance utilities tests..."
mpirun -n $NUM_PROCESSES python tests/test_performance_utils.py

# Run pipeline parallelism tests
echo "Running pipeline parallelism tests..."
mpirun -n $NUM_PROCESSES python tests/test_pipeline_parallel.py

# Run ZeRO optimizer tests
echo "Running ZeRO optimizer tests..."
mpirun -n $NUM_PROCESSES python tests/test_zero_optimizer.py

# Run tensor parallelism tests
echo "Running tensor parallelism tests..."
mpirun -n $NUM_PROCESSES python tests/test_tensor_parallel.py

# Run sequence parallelism tests
echo "Running sequence parallelism tests..."
mpirun -n $NUM_PROCESSES python tests/test_sequence_parallel.py

# Run activation checkpointing tests
echo "Running activation checkpointing tests..."
mpirun -n $NUM_PROCESSES python tests/test_activation_checkpoint.py

# Run expert parallelism tests
echo "Running expert parallelism tests..."
mpirun -n $NUM_PROCESSES python tests/test_expert_parallel.py

# Run mixture of experts model tests
echo "Running mixture of experts model tests..."
mpirun -n $NUM_PROCESSES python tests/test_moe_model.py

echo "All tests complete!" 