from functools import partial

import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv
from ..mamba.vmamba import VSSBlock, Permute


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    
    """
        in_channels,
        out_channels,
        kernel_size,
    """

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m

def post_mamba_block(vss_hidden_dim, ds_in_dim, ds_out_dim, norm_layer=nn.LayerNorm, channel_first=True):
    """
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
    """
    vss = VSSBlock(hidden_dim=vss_hidden_dim, drop_path=0.1)
    downsample = nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(ds_in_dim, ds_out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(ds_out_dim),
        )
    m = nn.Sequential(
        vss,
        downsample
    )
    return m

# def post_downsample(dim, out_dim, norm_layer=nn.LayerNorm, channel_first=False):
#     return nn.Sequential(
#             (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
#             nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
#             (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
#             norm_layer(out_dim),
#         )

def post_patch_embed(in_chans, embed_dim, patch_size, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=True):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )


# class SparseBasicBlock(spconv.SparseModule):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None):
#         super(SparseBasicBlock, self).__init__()

#         assert norm_fn is not None
#         if bias is None:
#             bias = norm_fn is not None
#         self.conv1 = spconv.SubMConv3d(
#             inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
#         )
#         self.bn1 = norm_fn(planes)
#         self.relu = nn.ReLU()
#         self.conv2 = spconv.SubMConv3d(
#             planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
#         )
#         self.bn2 = norm_fn(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = replace_feature(out, self.bn1(out.features))
#         out = replace_feature(out, self.relu(out.features))

#         out = self.conv2(out)
#         out = replace_feature(out, self.bn2(out.features))

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out = replace_feature(out, out.features + identity.features)
#         out = replace_feature(out, self.relu(out.features))

#         return out
def voxel_map_scatter(voxel_coords, voxel_features, sparse_shape, batch_size):
    pass
    


class VoxelBackBone8x_Mamba(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]  # nz, ny, nx = [41, 1600, 1408] 

        # self.conv_input = spconv.SparseSequential(
        #     spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
        #     norm_fn(16),
        #     nn.ReLU(),
        # )
        
        # block = post_act_block
        patch_embed = post_patch_embed
        vss = post_mamba_block
        # downsample = post_downsample

        self.vss_input = patch_embed(4, 16, 1)

        # self.conv1 = spconv.SparseSequential(
        #     block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        # )

        # self.conv2 = spconv.SparseSequential(
        #     # [1600, 1408, 41] <- [800, 704, 21]
        #     block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        #     block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        #     block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        # )

        # self.conv3 = spconv.SparseSequential(
        #     # [800, 704, 21] <- [400, 352, 11]
        #     block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        # )

        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        # )

        self.vss1 = vss(16, 16, 32)
        self.vss2 = vss(32, 32, 64)
        self.vss3 = vss(64, 64, 64)
        self.vss4 = vss(64, 64, 64)

        # last_pad = 0
        # last_pad = self.model_cfg.get('last_pad', last_pad)
        # self.conv_out = spconv.SparseSequential(
        #     # [200, 150, 5] -> [200, 150, 2]
        #     spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
        #                         bias=False, indice_key='spconv_down2'),
        #     norm_fn(128),
        #     nn.ReLU(),
        # )

        self.vss_out = patch_embed(64, 128, 1)

        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64,
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, x_idx, y_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        # x = self.conv_input(input_sp_tensor)
        x = self.vss_input(input_sp_tensor)

        # x_conv1 = self.conv1(x)
        # x_conv2 = self.conv2(x_conv1)
        # x_conv3 = self.conv3(x_conv2)
        # x_conv4 = self.conv4(x_conv3)

        x_vss1 = self.vss1(x)
        x_vss2 = self.vss2(x_vss1)
        x_vss3 = self.vss3(x_vss2)
        x_vss4 = self.vss4(x_vss3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        # out = self.conv_out(x_conv4)
        out = self.vss_out(x_vss4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_vss1,
                'x_conv2': x_vss2,
                'x_conv3': x_vss3,
                'x_conv4': x_vss4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


# class VoxelResBackBone8x(nn.Module):
#     def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
#         super().__init__()
#         self.model_cfg = model_cfg
#         use_bias = self.model_cfg.get('USE_BIAS', None)
#         norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

#         self.sparse_shape = grid_size[::-1] + [1, 0, 0]

#         self.conv_input = spconv.SparseSequential(
#             spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
#             norm_fn(16),
#             nn.ReLU(),
#         )
#         block = post_act_block

#         self.conv1 = spconv.SparseSequential(
#             SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
#             SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
#         )

#         self.conv2 = spconv.SparseSequential(
#             # [1600, 1408, 41] <- [800, 704, 21]
#             block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
#             SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
#             SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
#         )

#         self.conv3 = spconv.SparseSequential(
#             # [800, 704, 21] <- [400, 352, 11]
#             block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
#             SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
#             SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
#         )

#         self.conv4 = spconv.SparseSequential(
#             # [400, 352, 11] <- [200, 176, 5]
#             block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
#             SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
#             SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
#         )

#         last_pad = 0
#         last_pad = self.model_cfg.get('last_pad', last_pad)
#         self.conv_out = spconv.SparseSequential(
#             # [200, 150, 5] -> [200, 150, 2]
#             spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
#                                 bias=False, indice_key='spconv_down2'),
#             norm_fn(128),
#             nn.ReLU(),
#         )
#         self.num_point_features = 128
#         self.backbone_channels = {
#             'x_conv1': 16,
#             'x_conv2': 32,
#             'x_conv3': 64,
#             'x_conv4': 128
#         }

#     def forward(self, batch_dict):
#         """
#         Args:
#             batch_dict:
#                 batch_size: int
#                 vfe_features: (num_voxels, C)
#                 voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
#         Returns:
#             batch_dict:
#                 encoded_spconv_tensor: sparse tensor
#         """
#         voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
#         batch_size = batch_dict['batch_size']
#         input_sp_tensor = spconv.SparseConvTensor(
#             features=voxel_features,
#             indices=voxel_coords.int(),
#             spatial_shape=self.sparse_shape,
#             batch_size=batch_size
#         )
#         x = self.conv_input(input_sp_tensor)

#         x_conv1 = self.conv1(x)
#         x_conv2 = self.conv2(x_conv1)
#         x_conv3 = self.conv3(x_conv2)
#         x_conv4 = self.conv4(x_conv3)

#         # for detection head
#         # [200, 176, 5] -> [200, 176, 2]
#         out = self.conv_out(x_conv4)

#         batch_dict.update({
#             'encoded_spconv_tensor': out,
#             'encoded_spconv_tensor_stride': 8
#         })
#         batch_dict.update({
#             'multi_scale_3d_features': {
#                 'x_conv1': x_conv1,
#                 'x_conv2': x_conv2,
#                 'x_conv3': x_conv3,
#                 'x_conv4': x_conv4,
#             }
#         })

#         batch_dict.update({
#             'multi_scale_3d_strides': {
#                 'x_conv1': 1,
#                 'x_conv2': 2,
#                 'x_conv3': 4,
#                 'x_conv4': 8,
#             }
#         })
        
#         return batch_dict
