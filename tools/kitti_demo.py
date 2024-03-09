import _init_path
import os
import torch
import json

# try:
#     import open3d
#     from visual_utils import open3d_vis_utils as V
#     OPEN3D_FLAG = True
# except:
#     import mayavi.mlab as mlab
#     from visual_utils import visualize_utils as V
#     OPEN3D_FLAG = False

from pcdet.models import build_network, load_data_to_gpu
from pathlib import Path
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

# import mayavi.mlab as mlab
# from visual_utils import visualize_utils as V

def main(cfg_path, model_path, output_path, num_scenes, save_3d=False, save_roi=False, tag=None):
    cfg_from_yaml_file(cfg_path, cfg)
    logger = common_utils.create_logger()
    logger.info('-----------------< Creating data for visualization >-------------------------')
    kitti_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, root_path=Path(cfg.DATA_CONFIG.DATA_PATH),
    )
    
    logger.info(f'Total number of samples: \t{len(kitti_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=kitti_dataset)
    model.load_params_from_file(filename=model_path, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    # create folder for visualization
    # vis_path = '/'.join(os.path.normpath(model_path).split(os.path.sep)[:-2]) + '/visualization' + tag
    # os.makedirs(vis_path, exist_ok=True)

    with torch.no_grad():
        for idx, data_dict in enumerate(kitti_dataset):
            if idx >= num_scenes:
                break
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = kitti_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            
            if save_3d:
                # create folder for visualization
                print("[Kitti demo]: mkdir visualization")
                # vis_path = '/'.join(os.path.normpath(model_path).split(os.path.sep)[:-2]) + '/visualization' + tag
                vis_path = output_path
                os.makedirs(vis_path, exist_ok=True)
                # save
                torch.save(data_dict['points'][:,1:], os.path.join(vis_path, 'points_{}.pt'.format(int(data_dict['frame_id']))))
                torch.save(pred_dicts[0]['pred_boxes'], os.path.join(vis_path, 'pred_boxes_{}.pt'.format(int(data_dict['frame_id']))))
                torch.save(pred_dicts[0]['pred_scores'], os.path.join(vis_path, 'pred_scores_{}.pt'.format(int(data_dict['frame_id']))))
                torch.save(pred_dicts[0]['pred_labels'], os.path.join(vis_path, 'pred_labels_{}.pt'.format(int(data_dict['frame_id']))))
                torch.save(data_dict['gt_boxes'], os.path.join(vis_path, 'gt_boxes_{}.pt'.format(int(data_dict['frame_id']))))
                if 'gnn_edges_final' in pred_dicts[0]:
                    torch.save(pred_dicts[0]['gnn_edges_final'],os.path.join(vis_path, 'gnn_edges{}.pt'.format(int(data_dict['frame_id']))))
                    json.dump(pred_dicts[0]['edge_to_pred'] , open(os.path.join(vis_path, 'edge_to_predict{}.json'.format(int(data_dict['frame_id']))), 'w'))
                if save_roi:
                    pcl_roi = data_dict['pcl_roi']
                    mask_pclroi = pcl_roi[:, 0] != -1
                    pclroi_overlap = pcl_roi[mask_pclroi, 1:]
                    torch.save(pclroi_overlap, os.path.join(vis_path, 'roipcl_{}.pt'.format(int(data_dict['frame_id']))))
                    torch.save((data_dict['rois'].view(-1, data_dict['rois'].shape[-1]))[:, :7], os.path.join(vis_path, 'roibox_{}.pt'.format(int(data_dict['frame_id']))))
            else:
                # fig = V.draw_scenes(
                #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                # )
                # mlab.savefig(os.path.join(vis_path, 'points_{}.pt'.format(int(data_dict['frame_id']))))
                # pass
                # V.draw_scenes(
                #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                # )
                # if not OPEN3D_FLAG:
                #     mlab.show(stop=True)
                pass

    logger.info('Demo done.')

if __name__ == '__main__':

    output_path = "../output/vis/kitti/pv-rcnn-relation-ip/all_class/20240220/ep80"
    model_path = "../output/kitti_models/pv_rcnn_relation/train-AllClass-k16-Instance/20240220-144507/ckpt/checkpoint_epoch_80.pth"
    cfg_path = "cfgs/kitti_models/pv_rcnn_relation.yaml"

    NUMBER_OF_SCENES = 10

    # main(cfg_path, full_model_path, data_path, save_3d=True, tag=tag)
    main(cfg_path, model_path, output_path, num_scenes=NUMBER_OF_SCENES, save_3d=False, save_roi=True)
