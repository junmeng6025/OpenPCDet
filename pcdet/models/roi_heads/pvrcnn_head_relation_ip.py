import torch.nn as nn
import numpy as np
import torch

from copy import deepcopy
from easydict import EasyDict as edict
from scipy.spatial.transform import Rotation as R
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate
from ...ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu

# INSTANCE_PROP = [ 
#     'cx', 'cy', 'cz', 'dx', 'dy', 'dz', 'heading_cos', 'heading_sin', 
#     'dist', 'alpha_cos', 'alpha_sin', 'nr_pts',
#     'min_x', 'min_y', 'min_z', 'min_refl',
#     'max_x', 'max_y', 'max_z', 'max_refl',
#     'mean_x', 'mean_y', 'mean_z', 'mean_refl',
#     'std_x', 'std_y', 'std_z', 'std_refl'
# ]  # elongation only for Waymo

INSTANCE_PROP = [ 
    'cx', 'cy', 'cz', 'dx', 'dy', 'dz', 'heading_cos', 'heading_sin',
    'min_x', 'min_y', 'min_z', 'min_refl',
    'max_x', 'max_y', 'max_z', 'max_refl',
    'mean_x', 'mean_y', 'mean_z', 'mean_refl',
    'std_x', 'std_y', 'std_z', 'std_refl'
]  # elongation only for Waymo

# EXPORT_PTSROI = True

class PVRCNNHeadRelation(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, object_relation_config=None, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        self.roi_grid_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=input_channels, config=self.model_cfg.ROI_GRID_POOL
        )

        self.ip_dict = edict({k: i for i, k in enumerate(INSTANCE_PROP)})

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * num_c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)
        
        if object_relation_config.NAME == 'GNN':
            self.skip_head = False
            if object_relation_config.GLOBAL_INFORMATION:
                initial_input_dim = object_relation_config.GLOBAL_INFORMATION.MLP_LAYERS[-1]
                if not object_relation_config.GLOBAL_INFORMATION.CONCATENATED:
                    initial_input_dim += self.model_cfg.SHARED_FC[-1]
            else:
                initial_input_dim = self.model_cfg.SHARED_FC[-1]
            
            if object_relation_config.SKIP_CONNECTION:
                self.head_input_channels = initial_input_dim + sum(object_relation_config.LAYERS)
            else:
                if len(object_relation_config.LAYERS) == 0:
                    self.head_input_channels = self.model_cfg.SHARED_FC[-1]
                else:
                    self.head_input_channels = object_relation_config.LAYERS[-1]
        elif object_relation_config.NAME == 'CGNLNet':
            # TODO: udpate this
            self.head_input_channels = self.model_cfg.SHARED_FC[-1] + 256 + 256
        elif object_relation_config.NAME == 'GNN_BADET':
            self.head_input_channels = 256
            self.skip_head = True
        elif object_relation_config.NAME == 'GNN_NEW':
            self.head_input_channels = 256
            self.skip_head = False
        else:
            raise NotImplementedError
        
        self.cls_layers = self.make_fc_layers(
            input_channels=self.head_input_channels, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=self.head_input_channels,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def extract_instance_properties(self, roi_boxes, pcl):
        """
        Args:
           roi_boxes: tensor, (num_rois, 7)
           pcl: tensor, (num_rawpts, 4)
        Returns:

        """
        ipd = self.ip_dict
        data = np.zeros((roi_boxes.shape[0], len(ipd)), dtype=np.float32)  # in CPU
        # data = roi_boxes.new_zeros((roi_boxes.shape[0], len(ipd)), dtype=torch.float32)  # in GPU
        roi_boxes_np = deepcopy(roi_boxes.cpu().numpy())

        roi_boxes_np[:, 6] = np.mod(roi_boxes_np[:, 6], 2*np.pi)

        data[:, ipd.cx] = roi_boxes_np[:, 0]
        data[:, ipd.cy] = roi_boxes_np[:, 1]
        data[:, ipd.cz] = roi_boxes_np[:, 2]
        data[:, ipd.dx] = roi_boxes_np[:, 3]
        data[:, ipd.dy] = roi_boxes_np[:, 4]
        data[:, ipd.dz] = roi_boxes_np[:, 5]

        data[:, ipd.heading_cos] = np.cos(roi_boxes_np[:, 6])
        data[:, ipd.heading_sin] = np.sin(roi_boxes_np[:, 6])
        
        # alpha = roi_boxes_np[:, 6] - np.arctan2(roi_boxes_np[:, 1], roi_boxes_np[:, 0])
        # data[:, ipd.alpha_cos] = np.cos(alpha)
        # data[:, ipd.alpha_sin] = np.sin(alpha)

        # data[:, ipd.dist] = np.linalg.norm(roi_boxes_np[:, :3], axis=1)

        # if EXPORT_PTSROI:
        #     pcl_roi = deepcopy(pcl.cpu().numpy())
        #     pcl_roi = torch.from_numpy(np.hstack((np.full((pcl_roi.shape[0], 1), -1), pcl_roi))).to(pcl.device)  # (num_raw_pts, 5)=(60585, 5) [roi_id=-1, x, y, z, refl]

        for i in range(roi_boxes_np.shape[0]):
            box = deepcopy(roi_boxes_np[i, :])
            mask = points_in_boxes_gpu(pcl[None, :, :3], torch.from_numpy(box[None, None, :]).cuda())
            mask = mask[0, :] == 0  # (num_raw_pts)=(60585) [T/F]
            
            pcl_in_box = deepcopy(pcl[mask, :].cpu().numpy())
            # if EXPORT_PTSROI:
            #     pcl_roi[mask, 0] = i
                
            if pcl_in_box.shape[0] == 0:
                continue

            # data[i, ipd.nr_pts] = pcl_in_box.shape[0]
            
            # move to center
            pcl_in_box[:, :3] -= box[:3]
            box[:3] = 0

            # rotate to align with x-axis
            rotmat = R.from_euler('z', box[6], degrees=False).as_matrix()
            
            pcl_in_box[:, :3] = np.matmul(pcl_in_box[:, :3], rotmat)
            box[6] = 0

            # scale to unit box
            pcl_in_box[:, :3] /= box[3:6]
            box[3:6] = 1

            data[i, ipd.min_x] = np.min(pcl_in_box[:, 0])
            data[i, ipd.min_y] = np.min(pcl_in_box[:, 1])
            data[i, ipd.min_z] = np.min(pcl_in_box[:, 2])
            data[i, ipd.min_refl] = np.min(pcl_in_box[:, 3])
            # data[i, ipd.min_elongation] = np.min(pcl_in_box[:, 4])  # elongation for Waymo, unavailiable for KITTY

            data[i, ipd.max_x] = np.max(pcl_in_box[:, 0])
            data[i, ipd.max_y] = np.max(pcl_in_box[:, 1])
            data[i, ipd.max_z] = np.max(pcl_in_box[:, 2])
            data[i, ipd.max_refl] = np.max(pcl_in_box[:, 3])
            # data[i, ipd.max_elongation] = np.max(pcl_in_box[:, 4])

            data[i, ipd.mean_x] = np.mean(pcl_in_box[:, 0])
            data[i, ipd.mean_y] = np.mean(pcl_in_box[:, 1])
            data[i, ipd.mean_z] = np.mean(pcl_in_box[:, 2])
            data[i, ipd.mean_refl] = np.mean(pcl_in_box[:, 3])
            # data[i, ipd.mean_elongation] = np.mean(pcl_in_box[:, 4])

            data[i, ipd.std_x] = np.std(pcl_in_box[:, 0])
            data[i, ipd.std_y] = np.std(pcl_in_box[:, 1])
            data[i, ipd.std_z] = np.std(pcl_in_box[:, 2])
            data[i, ipd.std_refl] = np.std(pcl_in_box[:, 3])
            # data[i, ipd.std_elongation] = np.std(pcl_in_box[:, 4])

            data_tensor = torch.from_numpy(data).to(roi_boxes.device)
        # roi_count = np.count_nonzero((pcl_roi[:, 0].cpu().numpy()) != -1)
        # print("Num roi in 'roi_boxes'=%d; Count in 'pcl_roi'=%d")%(roi_boxes_np.shape[0],roi_count)
        # return data_tensor, pcl_roi  # (num_roi, 28)
        return data_tensor  # (num_roi, 28)


    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                points: (num_rawpts, 5)  [bs_idx, x, y, z, refl]
                rois: (B, num_rois, 7 + C)
                roi_scores: (1, num_rois)
                point_coords: (num_keypoints, 4)  [bs_idx, x, y, z]
                point_features: (num_keypoints, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']  # (1, 100, 7)
        roi_scores = batch_dict['roi_scores']  # (1,100)
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        # point_part_offset = batch_dict['point_part_offset']

        roi_boxes = (rois.view(-1, rois.shape[-1]))[:, :7]  # (num_roi, 7)=(100, 7)
        pcl = batch_dict['points']  # (num_raw_pts, 5)=(60585, 5) [bs_idx, x, y, z, refl]
        # ip_features, pcl_roi = self.extract_instance_properties(roi_boxes, pcl[:, 1:])  # (num_roi, 28)=(100, 28) -> DEVICE should be 'CUDA'
        ip_features = self.extract_instance_properties(roi_boxes, pcl[:, 1:])  # (num_roi, 28)=(100, 28) -> DEVICE should be 'CUDA'
        # if EXPORT_PTSROI:
        #     batch_dict['pcl_roi'] = pcl_roi

        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)=(100, 216, 128)
        return pooled_features, ip_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = batch_dict.get('roi_targets_dict', None)
            if targets_dict is None:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']

        _, N, _ = batch_dict['rois'].shape
        # RoI aware pooling
        pooled_features, ip_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C) C=128; (BxN, 28)

        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6) C=128
        
        pooled_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))  # (BxN, C', 1)=(100, 256, 1)
        batch_dict['pooled_features'] = pooled_features.view(-1, N, self.model_cfg.SHARED_FC[-1])  # (1, 100, 256)
        batch_dict['ip_features'] = ip_features.view(-1, N, len(self.ip_dict))  # (1, 100, 28)

        self.forward_ret_dict = targets_dict

        
        # rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        # rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        # if not self.training:
        #     batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
        #         batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
        #     )
        #     batch_dict['batch_cls_preds'] = batch_cls_preds
        #     batch_dict['batch_box_preds'] = batch_box_preds
        #     batch_dict['cls_preds_normalized'] = False
        # else:
        #     targets_dict['rcnn_cls'] = rcnn_cls
        #     targets_dict['rcnn_reg'] = rcnn_reg

        #     self.forward_ret_dict = targets_dict

        return batch_dict

    def final_predictions(self, batch_dict):
        if self.skip_head:
            rcnn_cls = batch_dict['rcnn_cls']
            rcnn_reg = batch_dict['rcnn_reg']
        else:
            shared_features = batch_dict['related_features']  # (100, 1280)
            shared_features = shared_features.view(-1, self.head_input_channels, 1)  # (100, 1280, 1)
            rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)  (100, 1)
            rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)       (100, 7)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds  # (B, N, 1)=(1, 100, 1)
            batch_dict['batch_box_preds'] = batch_box_preds  # (B, N, 7)=(1, 100, 7)
            batch_dict['cls_preds_normalized'] = False
        else:
            self.forward_ret_dict['rcnn_cls'] = rcnn_cls
            self.forward_ret_dict['rcnn_reg'] = rcnn_reg

        return batch_dict

