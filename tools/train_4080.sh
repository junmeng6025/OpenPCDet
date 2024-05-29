# # 定义一个字符串数组
# CFG_LS=(
#     "cfgs/waymo_models/pv_rcnn_mamba.yaml" 
#     "cfgs/waymo_models/pv_rcnn_relation_mamba.yaml" 
#     "cfgs/waymo_models/PartA2_mamba.yaml" 
#     "cfgs/waymo_models/PartA2_relation_mamba.yaml" 
#     "cfgs/waymo_models/pv_rcnn.yaml" 
#     "cfgs/waymo_models/pv_rcnn_relation.yaml" 
#     "cfgs/waymo_models/PartA2.yaml" 
#     "cfgs/waymo_models/PartA2_relation.yaml" 
#     )
# TAG="train_mamba_waymo_bs2"

# # 使用循环逐个读取数组元素
# for CFG_PATH in "${CFG_LS[@]}"
# do
#     echo "[Train] Starting train.py ... Using cfg $CFG_PATH"
#     # printf "[Train] Starting train.py ... Using cfg ${CFG_PATH}"
#     python train.py --cfg_file ${CFG_PATH} --extra_tag ${TAG} # BS=2
# done

# "cfgs/waymo_models/pv_rcnn_mamba.yaml" 
# "cfgs/waymo_models/pv_rcnn_relation_mamba.yaml" 


CFG01="cfgs/waymo_models/pv_rcnn_mamba.yaml"
CFG02="cfgs/waymo_models/pv_rcnn_relation_mamba.yaml" 
CFG03="cfgs/waymo_models/PartA2_mamba.yaml" 
CFG04="cfgs/waymo_models/PartA2_relation_mamba.yaml" 
CFG05="cfgs/waymo_models/pv_rcnn.yaml" 
CFG06="cfgs/waymo_models/pv_rcnn_relation.yaml" 
CFG07="cfgs/waymo_models/PartA2.yaml" 
CFG08="cfgs/waymo_models/PartA2_relation.yaml" 

TAG="train_mamba_waymo_bs2"

echo "[Train] Starting train.py ... Using cfg $CFG01"
CKPT01="/root/OpenPCDet/output/waymo_models/pv_rcnn_mamba/train_mamba_waymo_bs2/20240524-160249/ckpt/checkpoint_epoch_2.pth"
python train.py --cfg_file ${CFG01} --extra_tag ${TAG} --ckpt ${CKPT01} --date_tag "20240524-160249"

echo "[Train] Starting train.py ... Using cfg $CFG02"
CKPT02="/root/OpenPCDet/output/waymo_models/pv_rcnn_relation_mamba/train_mamba_waymo_bs2/20240524-234853/ckpt/checkpoint_epoch_12.pth"
python train.py --cfg_file ${CFG02} --extra_tag ${TAG} --ckpt ${CKPT02} --date_tag "20240524-234853"

echo "[Train] Starting train.py ... Using cfg $CFG03"
CKPT03="/root/OpenPCDet/output/waymo_models/PartA2_mamba/train_mamba_waymo_bs2/20240526-164142/ckpt/checkpoint_epoch_6.pth"
python train.py --cfg_file ${CFG03} --extra_tag ${TAG} --ckpt ${CKPT03} --date_tag "20240526-164142"

echo "[Train] Starting train.py ... Using cfg $CFG04"
CKPT04="/root/OpenPCDet/output/waymo_models/PartA2_relation_mamba/train_mamba_waymo_bs2/20240527-030544/ckpt/checkpoint_epoch_19.pth"
python train.py --cfg_file ${CFG04} --extra_tag ${TAG} --ckpt ${CKPT04} --date_tag "20240527-030544"

