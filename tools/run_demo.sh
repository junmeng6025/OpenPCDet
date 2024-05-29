# DATA_PATH_HOST="data/kitti/PointCloud/velodyne_points/data/0000000094.bin"
# DATA_PATH_WORKSTATION="data/kitti/PointCloud/0008/000126.bin"

DATA_PATH="../data/kitti/training/velodyne/000126.bin"

# quick demo ================================================
# CFG_PATH="cfgs/kitti_models/second.yaml"
# MODEL_PATH="../models/demo/pv_rcnn_8369.pth"

# marc's pvrcnn-relation ====================================
CFG_PATH="cfgs/kitti_models/pv_rcnn_relation.yaml"
MODEL_PATH="../models/marc/kitti/pv-rcnn-relation/all_classes/2023-10-18_08-08-10/ckpt/checkpoint_epoch_80.pth"

# ===========================================================
printf "[Demo] Starting demo.py ..."

python demo.py \
    --cfg_file ${CFG_PATH} \
    --ckpt ${MODEL_PATH} \
    --data_path ${DATA_PATH}

# Pretrained Model
# https://drive.google.com/file/d/1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-