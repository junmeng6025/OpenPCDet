import _init_path
import argparse
import glob
from pathlib import Path

# try:
#     import open3d
#     from visual_utils import open3d_vis_utils as V
#     OPEN3D_FLAG = True
# except:
#     import mayavi.mlab as mlab
#     from visual_utils import visualize_utils as V
#     OPEN3D_FLAG = False

import mayavi.mlab as mlab
from visual_utils import visualize_utils as V
OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

def draw_scenes_roi(points, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None, roi_boxes=None, pcl_roi=None):
    if not isinstance(points, np.ndarray):
        points = points.cpu().numpy()
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.cpu().numpy()
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.cpu().numpy()
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.cpu().numpy()
    if roi_boxes is not None and not isinstance(roi_boxes, np.ndarray):
        roi_boxes = roi_boxes.cpu().numpy()
    if pcl_roi is not None and not isinstance(pcl_roi, np.ndarray):
        pcl_roi = pcl_roi.cpu().numpy()

    fig = V.visualize_pts(points)
    fig = V.draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))
    if gt_boxes is not None:
        corners3d = V.boxes_to_corners_3d(gt_boxes)
        fig = V.draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)

    if ref_boxes is not None and len(ref_boxes) > 0:
        ref_corners3d = V.boxes_to_corners_3d(ref_boxes)
        if ref_labels is None:
            fig = V.draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=ref_scores, max_num=100)
        else:
            for k in range(ref_labels.min(), ref_labels.max() + 1):
                cur_color = tuple(V.box_colormap[k % len(V.box_colormap)])
                mask = (ref_labels == k)
                fig = V.draw_corners3d(ref_corners3d[mask], fig=fig, color=cur_color, cls=ref_scores[mask], max_num=100)

    if roi_boxes is not None and len(roi_boxes) > 0:
        roi_corners3d = V.boxes_to_corners_3d(roi_boxes)
        if ref_labels is None:
            fig = V.draw_corners3d(roi_corners3d, fig=fig, color=(0, 1, 1), cls=ref_scores, max_num=100)
        else:
            for k in range(ref_labels.min(), ref_labels.max() + 1):
                cur_color = tuple(V.box_colormap[k % len(V.box_colormap)])
                mask = (ref_labels == k)
                fig = V.draw_corners3d(roi_corners3d[mask], fig=fig, color=cur_color, cls=ref_scores[mask], max_num=100)

    fig = V.visualize_pts(pcl_roi, fgcolor=(1, 0.5, 0))

    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    return fig


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="./cfgs/kitti_models/pv_rcnn_relation.yaml",
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default="../data/kitti/demo_pcl/000235.bin",
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default="../models/marc/kitti/pv-rcnn-relation/all_classes/2023-10-18_08-08-10/ckpt/checkpoint_epoch_80.pth", 
                        help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    print("=> Starting demo... Using %s\n"%('open3d' if OPEN3D_FLAG else 'mayavi'))
    logger.info('-----------------< Quick Demo of OpenPCDet >-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        DRAW_ROI = True
        DRAW_PRED = False
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            if DRAW_ROI:
                pcl_roi = data_dict['pcl_roi']
                mask_pclroi = pcl_roi[:, 0] != -1
                pclroi_overlap = pcl_roi[mask_pclroi, 1:]
                draw_scenes_roi(
                    points=data_dict['points'][:, 1:],
                    roi_boxes=data_dict['rois'].view(-1, data_dict['rois'].shape[-1])[:, :7],
                    pcl_roi=pclroi_overlap,
                )

            if DRAW_PRED:
                V.draw_scenes(
                    points=data_dict['points'][:, 1:],
                    ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'],
                    ref_labels=pred_dicts[0]['pred_labels'],
                )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
