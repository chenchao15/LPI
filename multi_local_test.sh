#!/bin/bash

classname=02691156
for k in {0..300}
do
  for i in {0..100}
  do
    python LPI.py \
        --data_dir data/ \
        --out_dir output/$k/ \
        --pattern_num 100 \
        --batch_size 1 \
        --gauss_value 1 \
        --start_index $k \
        --class_idx $classname \
        --dataset other \
        --test_type 1\
        --thresh 0.005\
        --test_local $i\
        --CUDA 6
  done
done
