DATA_PATH_HOST="data/kitti/PointCloud/velodyne_points/data/0000000094.bin"
DATA_PATH_WORKSTATION="data/kitti/PointCloud/0008/000126.bin"

printf "[Demo] Starting demo.py ..."

python tools/demo.py --cfg_file tools/cfgs/kitti_models/pv_rcnn.yaml \
    --ckpt models/pv_rcnn_8369.pth \
    --data_path ${DATA_PATH_HOST}

# Pretrained Model
# https://drive.google.com/file/d/1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-