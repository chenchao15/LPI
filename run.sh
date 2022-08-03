#!/bin/bash

classname=02691156
python LPI.py \
	--data_dir data/ \
        --pattern_num 100 \
        --batch_size 1 \
        --gauss_value 1 \
        --start_index 0 \
	--out_dir output/single_shape/ \
	--class_idx $classname \
	--train \
	--dataset other \
	--CUDA 6

python LPI.py \
        --data_dir data/ \
        --pattern_num 100 \
        --gauss_value 1 \
        --start_index 0 \
        --out_dir output/single_shape/ \
        --class_idx $classname \
        --dataset other \
        --CUDA 6
