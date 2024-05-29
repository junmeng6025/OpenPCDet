CFG_PATH="cfgs/kitti_models/PartA2_relation_mamba.yaml"
# PREV_CKPT="../output/kitti_models/pointpillar_fps/train_pillarmamba_bs2/20240509-213748/ckpt/checkpoint_epoch_77.pth"
TAG="train_PartA2_relation_mamba_bs2"

printf "[Train] Starting train.py ... Using cfg ${CFG_PATH}"

# Resume from a previous ckpt ========================================
# python train.py --cfg_file ${CFG_PATH} --extra_tag ${TAG} --ckpt ${PREV_CKPT} --date_tag "20240509-213748"

# Start a new training ===============================================
python train.py --cfg_file ${CFG_PATH} --extra_tag ${TAG} # BS=2