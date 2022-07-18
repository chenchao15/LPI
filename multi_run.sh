#!/bin/bash

classname=02691156
for i in {0..3}
do
    python LPI.py \
	--data_dir /data/cc/project/PUI/local_neuralpull_dataset/ \
        --pattern_num 100 \
        --batch_size 1 \
        --gauss_value 1 \
        --start_index $i \
	--out_dir /data/cc/neural_1018_1_$i/ \
	--class_idx $classname \
	--train \
	--dataset other \
	--CUDA 6

    python LPI.py \
        --data_dir /data/cc/project/PUI/local_neuralpull_dataset/ \
        --pattern_num 100 \
        --gauss_value 1 \
        --start_index $i \
        --out_dir /data/cc/neural_1018_1_$i/ \
        --class_idx $classname \
        --dataset other \
        --CUDA 6
done
