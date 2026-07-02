#!/bin/bash

dataset='pascal'
method='fine_unimatch_v2'
exp='dinov2_base'
split='92'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
pretrained_path=exp/MaskFactory/unimatch_v2/dinov2_base/MaskFactory/maskfactory_59.pth #pretrained_path=exp/SegRCDB/unimatch_v2/dinov2_base/SegRCDB/segrcdb_59.pth
save_path=exp/$dataset/$method/$exp/gt_92_MaskFactory
mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path --pretrained-path $pretrained_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/out.log