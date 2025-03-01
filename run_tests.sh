#!/bin/bash

# Set the number of processes for testing
NUM_PROCESSES=2

# Run all the tests with MPI
echo "Running tests with $NUM_PROCESSES processes..."

echo -e "\n===== Basic Configuration Tests ====="
mpirun -n $NUM_PROCESSES python tests/test_config.py

echo -e "\n===== Naive Model Parallelism Tests ====="
mpirun -n $NUM_PROCESSES python tests/test_naive_parallel.py

echo -e "\n===== Megatron Model Parallelism Tests ====="
mpirun -n $NUM_PROCESSES python tests/test_megatron_parallel.py

echo -e "\n===== Data Parallelism Tests ====="
mpirun -n $NUM_PROCESSES python tests/test_data_parallel.py

echo -e "\n===== Gradient Utilities Tests ====="
mpirun -n $NUM_PROCESSES python tests/test_gradient_utils.py

echo -e "\n===== Checkpoint Management Tests ====="
mpirun -n $NUM_PROCESSES python tests/test_checkpoint_utils.py

echo -e "\n===== Performance Monitoring Tests ====="
mpirun -n $NUM_PROCESSES python tests/test_performance_utils.py

echo -e "\n===== Pipeline Parallelism Tests ====="
mpirun -n $NUM_PROCESSES python tests/test_pipeline_parallel.py

echo -e "\n===== ZeRO Optimizer Tests ====="
mpirun -n $NUM_PROCESSES python tests/test_zero_optimizer.py

echo -e "\n===== Tensor Parallelism Tests ====="
mpirun -n $NUM_PROCESSES python tests/test_tensor_parallel.py

echo -e "\n===== Sequence Parallelism Tests ====="
mpirun -n $NUM_PROCESSES python tests/test_sequence_parallel.py

echo -e "\n===== Activation Checkpointing Tests ====="
mpirun -n $NUM_PROCESSES python tests/test_activation_checkpoint.py

echo -e "\n===== All tests completed! =====" 