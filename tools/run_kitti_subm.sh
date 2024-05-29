# Generates results in .txt files

CFG_PATH="cfgs/kitti_models/pv_rcnn_car_class_only.yaml"
MODEL_PATH="../models/marc/kitti/pv-rcnn/car_class_only/2023-09-25_06-53-23/ckpt/checkpoint_epoch_79.pth"

# export CUDA_VISIBLE_DEVICES=0
python test.py --cfg_file ${CFG_PATH} --batch_size 2 --ckpt ${MODEL_PATH} --extra_tag "restxt_test" --eval_tag "pvrcnn_car_20230925_ep80" --save_to_file
