CFG_PATH="cfgs/waymo_models/pv_rcnn_mamba.yaml"
CKPT_DIR="../output/waymo_models/pv_rcnn_mamba/train_mamba_waymo_bs2/20240531-082012/ckpt"

# export CUDA_VISIBLE_DEVICES=1
echo "<Status> Eval ${CFG_PATH} Starting..."
# python test.py --cfg_file ${CFG_PATH} --batch_size 2 --ckpt_dir ${CKPT_DIR} --eval_all --start_epoch 80 --extra_tag "eval_trained_kitti"
python test.py --cfg_file ${CFG_PATH} --batch_size 2 --ckpt_dir ${CKPT_DIR} --eval_all --start_epoch 30 --extra_tag "eval_waymo"
echo "<Status> Eval ${CFG_PATH} Done"
