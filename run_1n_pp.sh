#!/bin/bash

# Define the ranges
nproc_range=(2 4 6)
num_threads_range=(1 2 3 4)
num_interop_threads_range=(1 2 3 4)

# Iterate through the combinations
for nproc in "${nproc_range[@]}"; do
  for num_threads in "${num_threads_range[@]}"; do
    for num_interop_threads in "${num_interop_threads_range[@]}"; do
      # Run the command with the current combination
      command="torchrun --nnodes=1 --nproc-per-node=$nproc --node-rank=0 --master-addr=192.168.1.104 --master-port=50000 pipeline_deit.py --chunk_size 1 --num_threads $num_threads --num_interop_threads $num_interop_threads"
      
      echo "Running: $command"
      $command
      echo "----------------------------------------------------------------------------"
      # Add any additional logic or commands if needed after running the torchrun command
    done
  done
done
