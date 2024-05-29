CFG_PATH="cfgs/waymo_models/pv_rcnn_relation.yaml"
CKPT_DIR="../output/ckpt/pvrcnn_rel/2023-09-27_11-17-07/ckpt"

# export CUDA_VISIBLE_DEVICES=1
printf "<Status> Eval PVRCNN-Rel Starting..."
# python test.py --cfg_file ${CFG_PATH} --batch_size 2 --ckpt_dir ${CKPT_DIR} --eval_all --start_epoch 80 --extra_tag "eval_trained_kitti"
python test.py --cfg_file ${CFG_PATH} --batch_size 2 --ckpt_dir ${CKPT_DIR} --eval_all --start_epoch 80 --extra_tag "eval_trained_kitti"
printf "<Status> Eval PVRCNN-Rel Done"
