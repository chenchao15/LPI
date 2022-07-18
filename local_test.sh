#!/bin/bash

classname=02691156
for i in {0..100}
do
  python LPI.py \
        --data_dir /data/cc/project/PUI/local_neuralpull_dataset/ \
        --out_dir /data/cc/neural_1018_1_0/ \
        --pattern_num 100 \
        --batch_size 1 \
        --gauss_value 1 \
        --start_index 0 \
        --class_idx $classname \
        --dataset other \
        --test_type 1\
        --thresh 0.005\
        --test_local $i\
        --CUDA 6
done
