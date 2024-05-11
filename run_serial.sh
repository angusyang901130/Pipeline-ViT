#!/bin/bash

for num_threads in {1..4}; do
  for num_interop_threads in {1..4}; do
    echo "Running python serial_deit.py --num_threads $num_threads --num_interop_threads $num_interop_threads"
    python serial_deit.py --num_threads $num_threads --num_interop_threads $num_interop_threads
    echo "--------------------------------------"
  done
done
