# # marc's pvrcnn-relation ====================================
CFG_PATH="cfgs/kitti_models/pv_rcnn_relation.yaml"
MODEL_PATH="../output/kitti_models/pv_rcnn_relation/20240116-224216/default/ckpt/checkpoint_epoch_80.pth"

# ===========================================================
printf "[Test] Starting test.py ..."
python test.py --cfg_file ${CFG_PATH} --batch_size 2 --ckpt ${MODEL_PATH}
