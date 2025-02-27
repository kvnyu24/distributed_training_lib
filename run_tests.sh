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

echo -e "\n===== All tests completed! =====" 