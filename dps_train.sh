#!/bin/bash

dataset='MaskFactory' #SegRCDB
method='dps_maskfactory' # dps_segrcdb.py
exp='dinov2_base'

config=configs/${dataset}.yaml
labeled_id_path=SegRCDB/txt/train.txt #labeled_id_path=SegRCDB/txt/train.txt
save_path=exp/$dataset/$method/$exp/MaskFactory #save_path=exp/$dataset/$method/$exp/SegRCDB
mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/out.log