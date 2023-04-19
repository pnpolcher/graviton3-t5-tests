#!/bin/bash

# Graviton3(E) (e.g. c7g, c7gn and HPC7g instances) supports BF16 format for ML acceleration. This can be enabled in oneDNN by setting the below environment variable
export DNNL_DEFAULT_FPMATH_MODE=BF16

# Enable primitive caching to avoid the redundant primitive allocation
# latency overhead. Please note this caching feature increases the
# memory footprint. Tune this cache capacity to a lower value to
# reduce the additional memory requirement.
export LRU_CACHE_CAPACITY=1024

# Enable Transparent huge page allocations from PyTorch C10 allocator
export THP_MEM_ALLOC_ENABLE=1

# Make sure the openmp threads are distributed across all the processes for multi process applications to avoid over subscription for the vcpus. For example if there is a single application process, then num_processes should be set to '1' so that all the vcpus are assigned to it with one-to-one mapping to omp threads
num_vcpus=$(getconf _NPROCESSORS_ONLN)
num_processes=1
export OMP_NUM_THREADS=$((1 > ($num_vcpus/$num_processes) ? 1 : ($num_vcpus/$num_processes)))
export OMP_PROC_BIND=false
export OMP_PLACES=cores

cd pytorch
source .venv/bin/activate
../t5.py
deactivate
