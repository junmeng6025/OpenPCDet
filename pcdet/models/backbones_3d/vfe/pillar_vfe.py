import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate

import pickle
PATH="/root/OpenPCDet/output/pillar_mamba_pkl/bs4"

def save_to_pkl(data, fname, path=PATH):
    with open('%s/%s.pkl'%(path, fname), 'wb') as f:
        pickle.dump(data, f)


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)
        print(self.pfn_layers)

        self.voxel_x = voxel_size[0]  # voxel_size=[0.16, 0.16, 4]  point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        # print("DEBUG: PillarVFE init finished.")

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        """
        batch_dict
            bs: 2
            points: (num_pts, 5)
            voxels: (num_voxels, N_pt_per_voxel=32, 4) [x, y, z, refl]
            voxel_num_points: (num_voxels)
            voxel_coords (num_voxels, 4) [bs_idx, z=0, x, y]
        """
        # save_to_pkl(batch_dict, "batch_dict_bs%d"%batch_dict['batch_size'])
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)  # (num_voxels, 1, 3)
        f_cluster = voxel_features[:, :, :3] - points_mean  # (num_voxels, N_pts=32, 3): [x_diff, y_diff, z_diff]

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]  # (num_voxels, N_pts=32, 4+3+3=10)
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)  # (num_voxels, N_pts=32, 3)
        # save_to_pkl(features, "features_origin")

        voxel_count = features.shape[1]  # N_pts=32
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)  # find out padded pts
        # save_to_pkl(mask, "mask_rigin")
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)  # (num_voxels, N_pts=32, 1)
        # save_to_pkl(mask, "mask_unsqueezed")
        features *= mask
        # save_to_pkl(features, "features_masked")
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        # save_to_pkl(features, "features_final")
        batch_dict['pillar_features'] = features  # (num_voxels, dim_pfn=64)  [25993, 64]
        save_to_pkl(batch_dict, "batch_dict_pp_bs4")
        return batch_dict
