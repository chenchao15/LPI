#!/bin/bash

classname=02691156
python LPI.py \
        --data_dir /data/cc/project/PUI/local_neuralpull_dataset/ \
        --pattern_num 100 \
        --gauss_value 1 \
        --start_index 0 \
        --out_dir /data/cc/neural_1018_1_0/ \
        --class_idx $classname \
        --dataset other \
        --CUDA 6
