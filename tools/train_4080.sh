#!/bin/bash

# 定义一个字符串数组
CFG_LS=(
    "cfgs/waymo_models/pv_rcnn_mamba.yaml" 
    "cfgs/waymo_models/pv_rcnn_relation_mamba.yaml" 
    "cfgs/waymo_models/PartA2_mamba.yaml" 
    "cfgs/waymo_models/PartA2_relation_mamba.yaml" 
    "cfgs/waymo_models/pv_rcnn.yaml" 
    "cfgs/waymo_models/pv_rcnn_relation.yaml" 
    "cfgs/waymo_models/PartA2.yaml" 
    "cfgs/waymo_models/PartA2_relation.yaml" 
    )
TAG="train_mamba_waymo_bs2"

# 使用循环逐个读取数组元素
for CFG_PATH in "${CFG_LS[@]}"
do
    echo "[Train] Starting train.py ... Using cfg $CFG_PATH"
    # printf "[Train] Starting train.py ... Using cfg ${CFG_PATH}"
    python train.py --cfg_file ${CFG_PATH} --extra_tag ${TAG} # BS=2
done

# "cfgs/waymo_models/pv_rcnn_mamba.yaml" 
# "cfgs/waymo_models/pv_rcnn_relation_mamba.yaml" 