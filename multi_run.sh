#!/bin/bash

classname=02691156
for i in {0..300}
do
    python LPI.py \
	--data_dir data/ \
        --pattern_num 100 \
        --batch_size 1 \
        --gauss_value 1 \
        --start_index $i \
	--out_dir output/$i/ \
	--class_idx $classname \
	--train \
	--dataset other \
	--CUDA 6

    python LPI.py \
        --data_dir data/ \
        --pattern_num 100 \
        --gauss_value 1 \
        --start_index $i \
        --out_dir output/$i/ \
        --class_idx $classname \
        --dataset other \
        --CUDA 6
done
