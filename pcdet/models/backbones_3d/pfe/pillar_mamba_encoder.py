import math
import numpy as np
import torch
import torch.nn as nn

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from ....ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ....utils import common_utils
from ...mamba.vmamba import VSSBlock, Permute

import pickle
PATH="/root/OpenPCDet/output/pillar_mamba_pkl/pillar_fps"
def save_pkl(data, fname, path=PATH):
    with open('%s/%s.pkl'%(path, fname), 'wb') as f:
        pickle.dump(data, f)


def check_idx(idxs, lb, ub):
    for idx in idxs:
        if idx < lb:
            idx=lb
        elif idx > ub-1:
            idx=ub-1
        else:
            continue


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


def sample_points_with_roi(rois, points, sample_radius_with_roi, num_max_points_of_part=200000):
    """
    Args:
        rois: (M, 7 + C)
        points: (N, 3)
        sample_radius_with_roi:
        num_max_points_of_part:

    Returns:
        sampled_points: (N_out, 3)
    """
    if points.shape[0] < num_max_points_of_part:
        distance = (points[:, None, :] - rois[None, :, 0:3]).norm(dim=-1)
        min_dis, min_dis_roi_idx = distance.min(dim=-1)
        roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
        point_mask = min_dis < roi_max_dim + sample_radius_with_roi
    else:
        start_idx = 0
        point_mask_list = []
        while start_idx < points.shape[0]:
            distance = (points[start_idx:start_idx + num_max_points_of_part, None, :] - rois[None, :, 0:3]).norm(dim=-1)
            min_dis, min_dis_roi_idx = distance.min(dim=-1)
            roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
            cur_point_mask = min_dis < roi_max_dim + sample_radius_with_roi
            point_mask_list.append(cur_point_mask)
            start_idx += num_max_points_of_part
        point_mask = torch.cat(point_mask_list, dim=0)

    sampled_points = points[:1] if point_mask.sum() == 0 else points[point_mask, :]

    return sampled_points, point_mask


def sector_fps(points, num_sampled_points, num_sectors):
    """
    Args:
        points: (N, 3)
        num_sampled_points: int
        num_sectors: int

    Returns:
        sampled_points: (N_out, 3)
    """
    sector_size = np.pi * 2 / num_sectors
    point_angles = torch.atan2(points[:, 1], points[:, 0]) + np.pi
    sector_idx = (point_angles / sector_size).floor().clamp(min=0, max=num_sectors)
    xyz_points_list = []
    xyz_batch_cnt = []
    num_sampled_points_list = []
    for k in range(num_sectors):
        mask = (sector_idx == k)
        cur_num_points = mask.sum().item()
        if cur_num_points > 0:
            xyz_points_list.append(points[mask])
            xyz_batch_cnt.append(cur_num_points)
            ratio = cur_num_points / points.shape[0]
            num_sampled_points_list.append(
                min(cur_num_points, math.ceil(ratio * num_sampled_points))
            )

    if len(xyz_batch_cnt) == 0:
        xyz_points_list.append(points)
        xyz_batch_cnt.append(len(points))
        num_sampled_points_list.append(num_sampled_points)
        print(f'Warning: empty sector points detected in SectorFPS: points.shape={points.shape}')

    xyz = torch.cat(xyz_points_list, dim=0)
    xyz_batch_cnt = torch.tensor(xyz_batch_cnt, device=points.device).int()
    sampled_points_batch_cnt = torch.tensor(num_sampled_points_list, device=points.device).int()

    sampled_pt_idxs = pointnet2_stack_utils.stack_farthest_point_sample(
        xyz.contiguous(), xyz_batch_cnt, sampled_points_batch_cnt
    ).long()

    sampled_points = xyz[sampled_pt_idxs]

    return sampled_points


class PillarMambaEncoder(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size  # [sx, sy, sz] = [0.16, 0.16, 4]
        self.point_cloud_range = point_cloud_range  # [x1, y1, z1, x2, y2, z2] = [0, -39.68, -3, 69.12, 39.68, 1]
        # nx = (x2 - x1)/sx = 69.12/0.16 = 432
        # ny = (y2 - y1)/sy = (39.68 + 39.68)/0.16 = 496 

        SA_cfg = self.model_cfg.SA_LAYER

        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0
        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name in ['bev', 'raw_points']:
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR

            if SA_cfg[src_name].get('INPUT_CHANNELS', None) is None:
                input_channels = SA_cfg[src_name].MLPS[0][0] \
                    if isinstance(SA_cfg[src_name].MLPS[0], list) else SA_cfg[src_name].MLPS[0]
            else:
                input_channels = SA_cfg[src_name]['INPUT_CHANNELS']

            cur_layer, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
                input_channels=input_channels, config=SA_cfg[src_name]
            )
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)

            c_in += cur_num_c_out

        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            c_bev = num_bev_features
            c_in += c_bev

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            self.SA_rawpoints, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
                input_channels=num_rawpoint_features - 3, config=SA_cfg['raw_points']
            )

            c_in += cur_num_c_out

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in
        
        self.mamba_dim = self.model_cfg.MAMBA_FEA_DIM  # 64, fit with 2D backbone
        self.mamba_stride = self.model_cfg.MAMBA_STRIDE
        self.patch_embed = self.make_patch_embed(in_chans=self.num_point_features, embed_dim=self.mamba_dim, patch_size=self.mamba_stride, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False)
        self.vssm = VSSBlock(hidden_dim=self.mamba_dim, drop_path=0.1)
    
    @staticmethod
    def make_patch_embed(in_chans, embed_dim, patch_size, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # TODO: patch_size = 4 or 1 ?
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        """
        Args:
            keypoints: (N1 + N2 + ..., 4)
            bev_features: (B, C, H, W)
            batch_size:
            bev_stride:

        Returns:
            point_bev_features: (N1 + N2 + ..., C)
        """
        x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]

        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            bs_mask = (keypoints[:, 0] == k)

            cur_x_idxs = x_idxs[bs_mask]
            cur_y_idxs = y_idxs[bs_mask]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features)

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (N1 + N2 + ..., C)
        return point_bev_features
    
    def keypoint_feature_scatter(self, keypoints, kp_features, scatter_features, batch_size):
        """
        use voxel_coord (pillar_coord) as index to get corresponding keypoint features
        Args:
            keypoints: (N1 + N2 + ..., 4) [bs_idx, x, y, z]
            kp_features:
            bev_features: (B, C, H, W)
            batch_size:
        Returns:
            kp_to_voxel_map: (B, C, H, W)
        """
        
        x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = torch.floor(x_idxs).type(torch.long)  # 0-431 -> W = 432
        y_idxs = torch.floor(y_idxs).type(torch.long)  # 0-495 -> H = 496
        # print("x idxs: %d - %d"%(torch.min(x_idxs), torch.max(x_idxs)))
        # print("y idxs: %d - %d"%(torch.min(y_idxs), torch.max(y_idxs)))

        batch_scatter_kpt_feature = torch.zeros_like(scatter_features)  # [bs, C, H, W]
        batch_scatter_kpt_feature = batch_scatter_kpt_feature.permute(0, 3, 2, 1)  # [bs, W, H, C]
        # print("DEBUG: start mapping ...")
        for k in range(batch_size):
            bs_mask = (keypoints[:, 0] == k)
            # print("DEBUG: Batch %d: %d keypoints"%(k, bs_mask.sum().item()))
            num_kpt = bs_mask.sum().item()
            cur_x_idxs = x_idxs[bs_mask]
            cur_y_idxs = y_idxs[bs_mask]
            cur_batch_scatter_kpt_feature = batch_scatter_kpt_feature[k]
            cur_kp_features = kp_features[bs_mask, :]
            # assert kp_features.shape[-1] == cur_batch_scatter_kpt_feature.shape[-1]  # make sure feature dims of kp & pillar feasible

            # cur_batch_scatter_kpt_feature[cur_x_idxs, cur_y_idxs, :] = kp_features[bs_mask, :]

            for pt_id in range(num_kpt):
                if cur_x_idxs[pt_id] > 431:
                    cur_x_idxs[pt_id] = 431
                if cur_y_idxs[pt_id] > 495:
                    cur_y_idxs[pt_id] = 495
                cur_batch_scatter_kpt_feature[cur_x_idxs[pt_id], cur_y_idxs[pt_id], :] = cur_kp_features[pt_id]
            
        batch_scatter_kpt_feature = batch_scatter_kpt_feature.permute(0, 3, 2, 1)  # [bs, C, H, W]
        
        return batch_scatter_kpt_feature
    
    def keypoint_feature_scatter_v2(self, keypoints, kp_features, batch_size):
        """
        use voxel_coord (pillar_coord) as index to get corresponding keypoint features
        Args:
            keypoints: (N1 + N2 + ..., 4) [bs_idx, x, y, z]  x: 0 ~ 69.12; y: -39.68 ~ 39.68
            kp_features:
            bev_features: (B, C, H, W)
            batch_size:
        Returns:
            batch_scatter_kpt_feature: (B, C, H, W)
        """
        [rg_x1, rg_y1, rg_z1, rg_x2, rg_y2, rg_z2] = self.point_cloud_range
        [sz_x, sz_y, sz_z] = self.voxel_size
        nx = int(math.floor((rg_x2 - rg_x1)/sz_x))  # 432 -> W
        ny = int(math.floor((rg_y2 - rg_y1)/sz_y))  # 496 -> H
        fea_dim = self.num_point_features

        batch_scatter_kpt_feature = []
        for b_idx in range(batch_size):
            scatter_kpt_feature = torch.zeros(
                fea_dim,
                nx*ny,
                dtype=kp_features.dtype,
                device=kp_features.device
            )
            bs_mask = (keypoints[:, 0] == b_idx)
            cur_kp_coord = keypoints[bs_mask, :]  # [b_idx, x, y, z]
            # print("x kpt: %d ~ %d"%(torch.min(keypoints[:, 1]), torch.max(keypoints[:, 1])))   
            # print("y kpt: %d ~ %d"%(torch.min(keypoints[:, 2]), torch.max(keypoints[:, 2])))

            cur_x_idxs = (cur_kp_coord[:, 1] - rg_x1)/sz_x
            # check_idx(cur_x_idxs, lb=0, ub=nx)
            cur_x_idxs = torch.floor(cur_x_idxs).type(torch.int)
            
            cur_y_idxs = (cur_kp_coord[:, 2] - rg_y1)/sz_y
            # check_idx(cur_y_idxs, lb=0, ub=ny)
            cur_y_idxs = torch.floor(cur_y_idxs).type(torch.int)

            # print("x idxs: %d ~ %d"%(torch.min(cur_x_idxs), torch.max(cur_x_idxs)))
            # print("y idxs: %d ~ %d"%(torch.min(cur_y_idxs), torch.max(cur_y_idxs)))

            # flatten_idx = torch.add(cur_x_idxs, nx*cur_y_idxs)
            flatten_idx = cur_x_idxs + nx * cur_y_idxs
            flatten_idx.type(torch.long)

            cur_kp_features = kp_features[bs_mask, :]
            cur_kp_features = cur_kp_features.t()
            # assert scatter_kpt_feature.shape[0] == cur_kp_features.shape[0]  # fea dim feasible
            scatter_kpt_feature[:, flatten_idx] = cur_kp_features

            batch_scatter_kpt_feature.append(scatter_kpt_feature)

        batch_scatter_kpt_feature = torch.stack(batch_scatter_kpt_feature, dim=0)  # [B, C, H*W]
        batch_scatter_kpt_feature = batch_scatter_kpt_feature.view(batch_size, fea_dim, ny, nx)  # [B, C, H, W]

        return batch_scatter_kpt_feature

    
    def calc_mamba_feature(self, pillar_scatter_batch):
        pillar_mamba_embed = self.patch_embed(pillar_scatter_batch)  # (B, H/ps, W/ps, fea_dim) = (4, 124, 108, 96)
        pillar_mamba_feature = self.vssm(pillar_mamba_embed)
        pillar_mamba_feature = pillar_mamba_feature.permute(0, 3, 1, 2)  # (4, 96, 124, 108)

        return pillar_mamba_feature

    def sectorized_proposal_centric_sampling(self, roi_boxes, points):
        """
        Args:
            roi_boxes: (M, 7 + C)
            points: (N, 3)

        Returns:
            sampled_points: (N_out, 3)
        """

        sampled_points, _ = sample_points_with_roi(
            rois=roi_boxes, points=points,
            sample_radius_with_roi=self.model_cfg.SPC_SAMPLING.SAMPLE_RADIUS_WITH_ROI,
            num_max_points_of_part=self.model_cfg.SPC_SAMPLING.get('NUM_POINTS_OF_EACH_SAMPLE_PART', 200000)
        )
        sampled_points = sector_fps(
            points=sampled_points, num_sampled_points=self.model_cfg.NUM_KEYPOINTS,
            num_sectors=self.model_cfg.SPC_SAMPLING.NUM_SECTORS
        )
        return sampled_points

    def get_sampled_points(self, batch_dict):
        """
        Args:
            batch_dict:

        Returns:
            keypoints: (N1 + N2 + ..., 4), where 4 indicates [bs_idx, x, y, z]
        """
        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:4]
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(
                batch_dict['voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            if self.model_cfg.SAMPLE_METHOD == 'FPS':
                cur_pt_idxs = pointnet2_stack_utils.farthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                ).long()

                if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    times = int(self.model_cfg.NUM_KEYPOINTS / sampled_points.shape[1]) + 1
                    non_empty = cur_pt_idxs[0, :sampled_points.shape[1]]
                    cur_pt_idxs[0] = non_empty.repeat(times)[:self.model_cfg.NUM_KEYPOINTS]

                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

            elif self.model_cfg.SAMPLE_METHOD == 'SPC':
                cur_keypoints = self.sectorized_proposal_centric_sampling(
                    roi_boxes=batch_dict['rois'][bs_idx], points=sampled_points[0]
                )
                bs_idxs = cur_keypoints.new_ones(cur_keypoints.shape[0]) * bs_idx
                keypoints = torch.cat((bs_idxs[:, None], cur_keypoints), dim=1)
            else:
                raise NotImplementedError

            keypoints_list.append(keypoints)

        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3) or (N1 + N2 + ..., 4)
        if len(keypoints.shape) == 3:
            batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1, 1)
            keypoints = torch.cat((batch_idx.float(), keypoints.view(-1, 3)), dim=1)
        
        # batch_dict['key_points'] = keypoints
        # save_pkl(batch_dict, "kpt_bs4")
        return keypoints

    @staticmethod
    def aggregate_keypoint_features_from_one_source(
            batch_size, aggregate_func, xyz, xyz_features, xyz_bs_idxs, new_xyz, new_xyz_batch_cnt,
            filter_neighbors_with_roi=False, radius_of_neighbor=None, num_max_points_of_part=200000, rois=None
    ):
        """

        Args:
            aggregate_func:
            xyz: (N, 3)
            xyz_features: (N, C)
            xyz_bs_idxs: (N)
            new_xyz: (M, 3)
            new_xyz_batch_cnt: (batch_size), [N1, N2, ...]

            filter_neighbors_with_roi: True/False
            radius_of_neighbor: float
            num_max_points_of_part: int
            rois: (batch_size, num_rois, 7 + C)
        Returns:

        """
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        if filter_neighbors_with_roi:
            point_features = torch.cat((xyz, xyz_features), dim=-1) if xyz_features is not None else xyz
            point_features_list = []
            for bs_idx in range(batch_size):
                bs_mask = (xyz_bs_idxs == bs_idx)
                _, valid_mask = sample_points_with_roi(
                    rois=rois[bs_idx], points=xyz[bs_mask],
                    sample_radius_with_roi=radius_of_neighbor, num_max_points_of_part=num_max_points_of_part,
                )
                point_features_list.append(point_features[bs_mask][valid_mask])
                xyz_batch_cnt[bs_idx] = valid_mask.sum()

            valid_point_features = torch.cat(point_features_list, dim=0)
            xyz = valid_point_features[:, 0:3]
            xyz_features = valid_point_features[:, 3:] if xyz_features is not None else None
        else:
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (xyz_bs_idxs == bs_idx).sum()

        pooled_points, pooled_features = aggregate_func(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=xyz_features.contiguous(),
        )
        return pooled_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        keypoints = self.get_sampled_points(batch_dict)

        point_features_list = []
        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            point_bev_features = self.interpolate_from_bev_features(
                keypoints, batch_dict['scatter_features'], batch_dict['batch_size'],
                bev_stride=batch_dict['spatial_features_stride']
            )
            point_features_list.append(point_bev_features)  # (B*N=4*2048, 256)

        batch_size = batch_dict['batch_size']

        new_xyz = keypoints[:, 1:4].contiguous()
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
        for k in range(batch_size):
            new_xyz_batch_cnt[k] = (keypoints[:, 0] == k).sum()

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            raw_points = batch_dict['points']

            # filter_neighbors_with_roi=self.model_cfg.SA_LAYER['raw_points'].get('FILTER_NEIGHBOR_WITH_ROI', False)
            # radius_of_neighbor=self.model_cfg.SA_LAYER['raw_points'].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None)
            # rois=batch_dict.get('rois', None)

            pooled_features = self.aggregate_keypoint_features_from_one_source(
                batch_size=batch_size, aggregate_func=self.SA_rawpoints,
                xyz=raw_points[:, 1:4],
                xyz_features=raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None,
                xyz_bs_idxs=raw_points[:, 0],
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt,
                filter_neighbors_with_roi=self.model_cfg.SA_LAYER['raw_points'].get('FILTER_NEIGHBOR_WITH_ROI', False),
                radius_of_neighbor=self.model_cfg.SA_LAYER['raw_points'].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None),
                rois=batch_dict.get('rois', None)
            )
            point_features_list.append(pooled_features)  # (B*N=4*2048, 32)

        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
            cur_features = batch_dict['multi_scale_3d_features'][src_name].features.contiguous()

            xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4], downsample_times=self.downsample_times_map[src_name],
                voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range
            )

            pooled_features = self.aggregate_keypoint_features_from_one_source(
                batch_size=batch_size, aggregate_func=self.SA_layers[k],
                xyz=xyz.contiguous(), xyz_features=cur_features, xyz_bs_idxs=cur_coords[:, 0],
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt,
                filter_neighbors_with_roi=self.model_cfg.SA_LAYER[src_name].get('FILTER_NEIGHBOR_WITH_ROI', False),
                radius_of_neighbor=self.model_cfg.SA_LAYER[src_name].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None),
                rois=batch_dict.get('rois', None)
            )
            point_features_list.append(pooled_features)  # (B*N=4*2048, 32 | 64 | 128 | 128)

        point_features = torch.cat(point_features_list, dim=-1)

        batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])  # (8192, 32) = (B*N, keypoint_feature)
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))  # 32 -> 64

        batch_dict['point_features'] = point_features  # (BxN, C)
        batch_dict['point_coords'] = keypoints  # (BxN, 4)

        # scatter_kpt_feature = self.keypoint_feature_scatter(keypoints, point_features, batch_dict['scatter_features'], batch_dict['batch_size'])
        scatter_kpt_feature = self.keypoint_feature_scatter_v2(keypoints, point_features, batch_dict['batch_size'])
        pillar_keypoint_features = torch.add(batch_dict['scatter_features'], scatter_kpt_feature)
        batch_dict['pillar_keypoint_features'] = pillar_keypoint_features

        pillar_mamba_features = self.calc_mamba_feature(pillar_keypoint_features)
        batch_dict['spatial_features'] = pillar_mamba_features  # # (B, H/ps, W/ps, fea_dim) = (4, 124, 108, 96)
        
        return batch_dict
