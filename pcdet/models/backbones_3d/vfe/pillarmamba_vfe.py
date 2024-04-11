import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from copy import deepcopy

import ocnn
from ocnn.octree import Octree
from ocnn.octree.points import Points
import dwconv

from timm.models.layers import DropPath
import math
from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba, Block

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# from .vfe_template import VFETemplate
from vfe_template import VFETemplate
from typing import Optional, List

# class PFNLayer(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  use_norm=True,
#                  last_layer=False):
#         super().__init__()
        
#         self.last_vfe = last_layer
#         self.use_norm = use_norm
#         if not self.last_vfe:
#             out_channels = out_channels // 2

#         if self.use_norm:
#             self.linear = nn.Linear(in_channels, out_channels, bias=False)
#             self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
#         else:
#             self.linear = nn.Linear(in_channels, out_channels, bias=True)

#         self.part = 50000

#     def forward(self, inputs):
#         if inputs.shape[0] > self.part:
#             # nn.Linear performs randomly when batch size is too large
#             num_parts = inputs.shape[0] // self.part
#             part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
#                                for num_part in range(num_parts+1)]
#             x = torch.cat(part_linear_out, dim=0)
#         else:
#             x = self.linear(inputs)
#         torch.backends.cudnn.enabled = False
#         x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
#         torch.backends.cudnn.enabled = True
#         x = F.relu(x)
#         x_max = torch.max(x, dim=1, keepdim=True)[0]

#         if self.last_vfe:
#             return x_max
#         else:
#             x_repeat = x_max.repeat(1, inputs.shape[1], 1)
#             x_concatenated = torch.cat([x, x_repeat], dim=2)
#             return x_concatenated


# ===================================================================================
class OctreeT(Octree):
    def __init__(self, octree: Octree, patch_size: int = 24, dilation: int = 4,
                nempty: bool = True, max_depth: Optional[int] = None,
                start_depth: Optional[int] = 2, **kwargs):
        super().__init__(octree.depth, octree.full_depth)
        self.__dict__.update(octree.__dict__)

        # self.patch_size = patch_size
        # self.dilation = dilation  # TODO dilation as a list
        self.nempty = nempty
        self.max_depth = max_depth or self.depth
        self.start_depth = self.full_depth or start_depth # if the value in the front is None, then get the latter one
        self.invalid_mask_value = -1e3
        assert self.start_depth > 1

        self.block_num = patch_size * dilation
        self.nnum_t = self.nnum_nempty if nempty else self.nnum
        self.nnum_a = ((self.nnum_t / self.block_num).ceil() * self.block_num).int()

        num = self.max_depth + 1
        self.batch_idx = [None] * num
        self.build_t()

    def build_t(self):
        for d in range(self.start_depth, self.max_depth + 1):
            self.build_batch_idx(d)

    def build_batch_idx(self, depth: int):
        batch = self.batch_id(depth, self.nempty)
        self.batch_idx[depth] = self.patch_partition(batch, depth, self.batch_size)

    def patch_partition(self, data: torch.Tensor, depth: int, fill_value=0):
        # num = self.nnum_a[depth] - self.nnum_t[depth]  # Err: Negative dim
        num =  self.nnum_t[depth] - self.nnum_a[depth]
        tail = data.new_full((num,) + data.shape[1:], fill_value)


class MLP(torch.nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                out_features: Optional[int] = None, activation=torch.nn.GELU,
                drop: float = 0.0, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features

        self.fc1 = torch.nn.Linear(self.in_features, self.hidden_features)
        self.act = activation()
        self.fc2 = torch.nn.Linear(self.hidden_features, self.out_features)
        self.drop = torch.nn.Dropout(drop, inplace=True)

    def forward(self, data: torch.Tensor):
        data = self.fc1(data)
        data = self.act(data)
        data = self.drop(data)
        data = self.fc2(data)
        data = self.drop(data)
        return data
    

class OctreeDWConvBn(torch.nn.Module):
    def __init__(self, in_channels: int, kernel_size: List[int] = [3],
                stride: int = 1, nempty: bool = False):
        super().__init__()
        self.conv = dwconv.OctreeDWConv(
            in_channels, kernel_size, nempty, use_bias=False)
        self.bn = torch.nn.BatchNorm1d(in_channels)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.conv(data, octree, depth)
        out = self.bn(out)
        return out
    

class RPE(torch.nn.Module):
    def __init__(self, patch_size: int, num_heads: int, dilation: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation
        self.pos_bnd = self.get_pos_bnd(patch_size)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3*self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def get_pos_bnd(self, patch_size: int):
        return int(0.8 * patch_size * self.dilation**0.5)

    def xyz2idx(self, xyz: torch.Tensor):
        mul = torch.arange(3, device=xyz.device) * self.rpe_num
        xyz = xyz.clamp(-self.pos_bnd, self.pos_bnd)
        idx = xyz + (self.pos_bnd + mul)
        return idx

    def forward(self, xyz):
        idx = self.xyz2idx(xyz)
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out

    def extra_repr(self) -> str:
        return 'num_heads={}, pos_bnd={}, dilation={}'.format(
                self.num_heads, self.pos_bnd, self.dilation)  # noqa


    def extra_repr(self) -> str:
        return 'dim={}, patch_size={}, num_heads={}, dilation={}'.format(
                self.dim)  # noqa


class OctreeMamba(nn.Module):
    def __init__(self, dim: int,
                    proj_drop: float = 0.0,):
        super().__init__()
        self.dim = dim
        
        self.pim = PointMambaMix(input_dim=dim, output_dim=dim,fused_add_norm=True)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        data = data.unsqueeze(0)
        # data = data.view(-1, K, C)#N,K,C->256,32,256 
        
        # data = data.permute(1, 0, 2)#B,N,C
        data = self.pim(data)
        data = data.squeeze(0)
        data = self.proj(data)
        data = self.proj_drop(data)
        return data
    
    def extra_repr(self) -> str:
        return 'dim={}'.format(
                self.dim)  # noqa


class PointMambaBlock(torch.nn.Module):
    def __init__(self, dim: int,
                proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
                activation: torch.nn.Module = torch.nn.GELU, **kwargs):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.mamba = OctreeMamba(dim,proj_drop)
        self.drop_path = ocnn.nn.OctreeDropPath(drop_path, nempty)
        self.cpe = OctreeDWConvBn(dim, nempty=nempty)

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        data = self.cpe(data, octree, depth) + data
        attn = self.mamba(self.norm1(data), octree, depth)
        data = data + self.drop_path(attn, octree, depth)
        return data
    

class PointMambaStage(torch.nn.Module):

  def __init__(self, dim: int, 
               proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
               activation: torch.nn.Module = torch.nn.GELU, interval: int = 6,
               use_checkpoint: bool = True, num_blocks: int = 2,
               pim_block=PointMambaBlock, **kwargs):
    super().__init__()
    self.num_blocks = num_blocks
    self.use_checkpoint = use_checkpoint
    self.interval = interval  # normalization interval
    self.num_norms = (num_blocks - 1) // self.interval

    self.blocks = torch.nn.ModuleList([pim_block(
        dim=dim
        , proj_drop=proj_drop,
        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
        nempty=nempty, activation=activation) for i in range(num_blocks)])
    # self.norms = torch.nn.ModuleList([
    #     torch.nn.BatchNorm1d(dim) for _ in range(self.num_norms)])

  def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
    for i in range(self.num_blocks):
      if self.use_checkpoint and self.training:
        data = checkpoint(self.blocks[i], data, octree, depth)
      else:
        data = self.blocks[i](data, octree, depth)
      # if i % self.interval == 0 and i != 0:
      #   data = self.norms[(i - 1) // self.interval](data)
    return data


class PatchEmbed(torch.nn.Module):

    def __init__(self, in_channels: int = 3, dim: int = 96, num_down: int = 2,
                nempty: bool = True, **kwargs):
        super().__init__()
        self.num_stages = num_down
        self.delta_depth = -num_down
        channels = [int(dim * 2**i) for i in range(-self.num_stages, 1)]

        self.convs = torch.nn.ModuleList([ocnn.modules.OctreeConvBnRelu(
            in_channels if i == 0 else channels[i], channels[i], kernel_size=[3],
            stride=1, nempty=nempty) for i in range(self.num_stages)])
        self.downsamples = torch.nn.ModuleList([ocnn.modules.OctreeConvBnRelu(
            channels[i], channels[i+1], kernel_size=[2], stride=2, nempty=nempty)
            for i in range(self.num_stages)])
        self.proj = ocnn.modules.OctreeConvBnRelu(
            channels[-1], dim, kernel_size=[3], stride=1, nempty=nempty)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        for i in range(self.num_stages):
            depth_i = depth - i
            data = self.convs[i](data, octree, depth_i)
            data = self.downsamples[i](data, octree, depth_i)
        data = self.proj(data, octree, depth_i - 1)
        return data


class Downsample(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                kernel_size: List[int] = [2], nempty: bool = True):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(out_channels)
        self.conv = ocnn.nn.OctreeConv(in_channels, out_channels, kernel_size,
                                    stride=2, nempty=nempty, use_bias=True)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        data = self.conv(data, octree, depth)
        data = self.norm(data)
        return data


class PointMamba(torch.nn.Module):
    def __init__(self, in_channels: int,
                channels: List[int],
                num_blocks: List[int],
                #  num_heads: List[int] = [6, 12, 24, 24],
                drop_path: float = 0.5,
                nempty: bool = True, stem_down: int = 2, **kwargs):
        super().__init__()
        self.nempty = nempty
        self.num_stages = len(num_blocks)
        self.stem_down = stem_down
        drop_ratio = torch.linspace(0, drop_path, sum(num_blocks)).tolist()

        self.patch_embed = PatchEmbed(in_channels, channels[0], stem_down, nempty)
        self.layers = torch.nn.ModuleList([PointMambaStage(
            dim=channels[i],
            drop_path=drop_ratio[sum(num_blocks[:i]):sum(num_blocks[:i+1])],
            nempty=nempty, num_blocks=num_blocks[i],)
            for i in range(self.num_stages)])
        self.downsamples = torch.nn.ModuleList([Downsample(
            channels[i], channels[i + 1], kernel_size=[2],
            nempty=nempty) for i in range(self.num_stages - 1)])

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        data = self.patch_embed(data, octree, depth)
        depth = depth - self.stem_down   # current octree depth
        # octree = OctreeT(octree, self.patch_size, self.dilation, self.nempty,
        #                  max_depth=depth, start_depth=depth-self.num_stages+1)
        octree = OctreeT(octree, self.nempty,
                        max_depth=depth, start_depth=depth-self.num_stages+1)
        features = {}
        for i in range(self.num_stages):
            depth_i = depth - i
            data = self.layers[i](data, octree, depth_i)
            features[depth_i] = data
            if i < self.num_stages - 1:
                data = self.downsamples[i](data, octree, depth_i)
        return features


# ====================================================
class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.0
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        # drop path 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    bimamba_type="none",
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    
    # mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, **ssm_cfg, **factory_kwargs)#创建Mamba类的实例
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)#创建Mamba类的实例
    
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
 
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:

                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class PointMambaMix(nn.Module):
    def __init__(self,  
        output_dim=512,
        input_dim=512,
        drop_path = 0.1,
        drop_out_in_block= 0.1,
        n_layer=1,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=True,
        device=None,
        dtype=None,
        bimamba_type="v2",
        **kwargs)->None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.input_dim = input_dim
        self.output_dim = output_dim 
        
        
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    input_dim,#嵌入x
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba_type=bimamba_type,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            input_dim, eps=norm_epsilon, **factory_kwargs
        )
        
        self.pre_logits = nn.Identity()
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }
        

    def forward_features(self, input_ids, inference_params=None):
        # hidden_states = self.embedding(input_ids)
        hidden_states = input_ids
        
        # print('input_ids.shape',input_ids.shape)
        residual = None
        
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn

            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
            

        #offset
        hidden_states = hidden_states - input_ids

        return hidden_states
    
    def forward(self, input_ids, inference_params=None):
        return input_ids


class PCLMamba(VFETemplate):
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.mamba_cfg=model_cfg.MAMBA
        self.octree_cfg=model_cfg.OCTREE
        assert len(self.mamba_cfg.CHANNELS)==len(self.mamba_cfg.NUM_BLOCKS)
        self.octree_merge = False
        self.device='cuda:0'

        self.ptmamba = PointMamba(
            in_channels=self.mamba_cfg.IN_CHN, 
            channels=self.mamba_cfg.CHANNELS, 
            num_blocks=self.mamba_cfg.NUM_BLOCKS,
            nempty=False).to(self.device)
        
    def get_output_feature_dim(self):
        return self.ptmamba.channels[-1]
        
    def get_input_feature(self, octree):
        """
        N: normal signal is extracted (3 channels)
        D: local displacement is extracted (1 channels)
        L: local coordinates of the averaged points in each octree node is extracted (3 channels)
        P: global coordinates are extracted (3 channels)
        F: other features (like colors) are extracted (k channels)
        """
        octree_feature=ocnn.modules.InputFeature(
            self.octree_cfg.FEATURES, 
            self.octree_cfg.NEMPTY)
        data = octree_feature(octree)
        return data
    
    def process_pcl(self, pcl_xyz):
        assert pcl_xyz.dim()==2
        assert pcl_xyz.shape[1]==3
        pts = Points(pcl_xyz, features=pcl_xyz, normals=pcl_xyz)
        # pts = Points(pcl_xyz, features=pcl_xyz)
        pts.cuda(non_blocking=True)
        octree = ocnn.octree.Octree(
            depth=self.octree_cfg.DEPTH, 
            full_depth=self.octree_cfg.FULL_DEPTH,
            device=pts.device)
        octree.build_octree(pts)
        # octree = points2octree(pts)
        octree.cuda(non_blocking=True)
        octree.construct_all_neigh()
        return octree, pts
    
    def forward(self, batch_dict):
        bs=batch_dict['batch_size']
        points = deepcopy(batch_dict['points'])
        points_batch = []
        for bs_id in range(bs):
            points_batch.append(points[points[:, 0]==bs_id, :])
            print("points num_batch%d: %d"%(bs_id, points_batch[bs_id].shape[0]))  # 18616; 20024

        if self.octree_merge:
            mamba_dict = {}
            points_ocnn = [Points(pcl[:, 1:4], features=pcl[:, 1:4], normals=pcl[:, 1:4]) for pcl in points_batch]
            octrees = [self.points2octree(pts) for pts in points_ocnn]
            octree = ocnn.octree.merge_octrees(octrees)
            octree.construct_all_neigh()
            points_ocnn = ocnn.octree.merge_points(points_ocnn)
            mamba_dict['octree'] = octree
            mamba_dict['pts_ocnn'] = points_ocnn
            data = self.get_input_feature(octree)
            mamba_features = self.ptmamba(data, octree, octree.depth)
            mamba_dict['mamba_features'] = mamba_features
            batch_dict['mamba_dict_merged'] = mamba_dict
        else:
            mamba_dict_batch = []
            for pcl in points_batch:
                pcl_xyz = deepcopy(pcl[:, 1:4])
                octree, pts = self.process_pcl(pcl_xyz)
                data = self.get_input_feature(octree)  # (79808, 4)
                mamba_features = self.ptmamba(data, octree, octree.depth)  # {'stage_idx': Tensor}
                mamba_dict = {
                    'octree': octree,
                    'pts_ocnn': pts,
                    'mamba_features': mamba_features,}  # (4096, 96) (4032, 96) (4096, 96) (4096, 96)
                mamba_dict_batch.append(mamba_dict)
            batch_dict['mamba_dicts'] = mamba_dict_batch
        return batch_dict


    


if __name__ == '__main__':
    import pickle
    PATH="/root/OpenPCDet/output/pillar_mamba_pkl/bs4"
    def save_pkl(data, fname, path=PATH):
        with open('%s/%s.pkl'%(path, fname), 'wb') as f:
            pickle.dump(data, f)

    def load_pkl(fname, path=PATH):
        with open('%s/%s.pkl'%(path, fname), 'rb') as f:
            return pickle.load(f)
    
    from easydict import EasyDict as edict
    cfg_pclmamba = edict({
        'MAMBA':{
            'IN_CHN': 4,
            'CHANNELS': [96],
            'NUM_BLOCKS': [2],
            'NEMPTY': False,
        },
        'OCTREE':{
            'DEPTH': 6,
            'FULL_DEPTH': 2,
            'NEMPTY': False,
            'FEATURES': 'ND',
            'ORIENT_NORM': 'xyz',
        },
    })
    # # pillar feature analysis
    # features_origin = load_pkl("features_origin")  # (9913, 32, 10)
    # mask_rigin = load_pkl("mask_rigin")            # (9913, 32)
    # mask_unsqueezed = load_pkl("mask_unsqueezed")  # (9913, 32, 1)
    # features_masked = load_pkl("features_masked")  # (9913, 32, 10)  PFN: in_features=10
    # features_final = load_pkl("features_final")    # (9913, 64)      PFN: out_features=64
    # print("DEBUG: pkl loaded")

    # batch_dict = load_pkl("batch_dict_bs32")
    batch_dict = load_pkl("batch_dict_bs4")
    """
    batch_dict
        bs: 2
        points: (num_pts, 5)
        voxels: (num_voxels, N=32, 4) [x, y, z, refl],  N is num pts per voxel
        voxel_num_points: (num_voxels)
        voxel_coords: (num_voxels, 4) [bs_idx, z=0, x, y]
        points: (num_pts, 5) [bs_idx, x, y, z, refl]
    """
    B=batch_dict['batch_size']
    N=batch_dict['voxels'].shape[1]  # N pts per voxel
    N_PTS=batch_dict['points'].shape[0]
    N_VOX=batch_dict['voxels'].shape[0]

    # voxels, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']

    # voxel_batch0=coords[coords[:, 0]==0, :]
    # print("Voxel num_batch0: %d"%voxel_batch0.shape[0])  # 6812
    # voxel_batch1=coords[coords[:, 0]==1, :]
    # print("Voxel num_batch1: %d"%voxel_batch1.shape[0])  # 3101

    # DEBUG: Use PCLMamba =====================================================
    pclmamba = PCLMamba(model_cfg=cfg_pclmamba)
    pclmamba(batch_dict)
    # save_pkl(batch_dict, "batch_dict_pm_bs%d"%batch_dict['batch_size'])

    print("DEBUG: End Debug")







    # DEBUG: code =====================================================
    points = batch_dict['points']
    points_batch = []
    for bs_id in range(B):
        points_batch.append(points[points[:, 0]==bs_id, :])
        print("points num_batch%d: %d"%(bs_id, points_batch[bs_id].shape[0]))  # 18616; 20024

    # CFG: ptmamba
    IN_CHN=4
    CHANNELS=[96]
    NUM_BLOCKS=[2]
    # CHANNELS=[96, 192, 384, 384]
    # NUM_BLOCKS=[2, 2, 18, 2]
    
    assert len(CHANNELS)==len(NUM_BLOCKS) # ==num_stages
    ptmamba = PointMamba(in_channels=IN_CHN, channels=CHANNELS, num_blocks=NUM_BLOCKS, nempty=False)
    ptmamba = ptmamba.to(points.device)
    print("DEBUG: ptmamba init:\n")
    print(ptmamba)

    octree_feature = ocnn.modules.InputFeature(feature='ND', nempty=False)
    """
    N: normal signal is extracted (3 channels)
    D: local displacement is extracted (1 channels)
    L: local coordinates of the averaged points in each octree node is extracted (3 channels)
    P: global coordinates are extracted (3 channels)
    F: other features (like colors) are extracted (k channels)
    """

    # CFG: octree
    DEPTH=6
    FULL_DEPTH=2
    ORIENT_NORM='xyz'

    # def points2octree(points):
    #     device=points.device
    #     print("DEBUG: Using device %s"%device)
    #     octree = ocnn.octree.Octree(depth=6, full_depth=2, device=device)
    #     octree.build_octree(points)
    #     return octree
    
    # def process_voxels(voxels):
    #     points = [Points(pts_voxel[:, :3]) for pts_voxel in torch.unbind(voxels, dim=0)]
    #     octrees = [points2octree(pts) for pts in points]
    #     octrees_merged = ocnn.octree.merge_octrees(octrees)
    #     octrees_merged.construct_all_neigh()
    #     points_merged = ocnn.octree.merge_points(points)
    #     return octrees_merged, points_merged
    
    # def process_voxel(pts_voxel):
    #     pts = Points(pts_voxel[:, :3])
    #     octree = points2octree(pts)
    #     octree.construct_all_neigh()
    #     return octree, pts

    # mamba_dicts = []
    # for pts_voxel in torch.unbind(voxels, dim=0):
    #     octree, pts = process_voxel(pts_voxel)
    #     data = octree_feature(octree)
    #     logits = ptmamba(data, octree, octree.depth)
    #     mamba_dict = {
    #         'octree': octree,
    #         'pts': pts,
    #         'logits': logits,}
    #     mamba_dicts.append(mamba_dict)

    # def points2octree(points):
    #     device=points.device
    #     print("DEBUG: octree gen using device %s"%device)
    #     octree = ocnn.octree.Octree(depth=6, full_depth=2, device=device)
    #     octree.build_octree(points)
    #     return octree

    def process_pcl(pcl_xyz):
        assert pcl_xyz.dim()==2
        assert pcl_xyz.shape[1]==3
        pts = Points(pcl_xyz, features=pcl_xyz, normals=pcl_xyz)
        # pts = Points(pcl_xyz, features=pcl_xyz)
        pts.cuda(non_blocking=True)
        octree = ocnn.octree.Octree(depth=6, full_depth=2, device=pts.device)
        octree.build_octree(pts)
        # octree = points2octree(pts)
        octree.cuda(non_blocking=True)
        octree.construct_all_neigh()
        return octree, pts
    
    # Debug: Batch merge ============================================================================
    # points_ocnn = [Points(pcl[:, 1:4], features=pcl[:, 1:4], normals=pcl[:, 1:4]) for pcl in points_batch]
    # octrees = [points2octree(pts) for pts in points_ocnn]
    # octree = ocnn.octree.merge_octrees(octrees)
    # octree.construct_all_neigh()
    # points_ocnn = ocnn.octree.merge_points(points_ocnn)
    # # mamba_dict = {
    # #     'points_ocnn': points_ocnn,
    # #     'octree': octree,
    # # }
    # data = octree_feature(octree)
    # logits = ptmamba(data, octree, octree.depth)

    # Debug: NO batch merge ============================================================================
    mamba_dict_batch = []
    for pcl in points_batch:
        pcl_xyz = deepcopy(pcl[:, 1:4])
        octree, pts = process_pcl(pcl_xyz)
        data = octree_feature(octree)  # (79808, 4)
        mamba_features = ptmamba(data, octree, octree.depth)  # {'stage_idx': Tensor}
        mamba_dict = {
            'octree': octree,
            'pts': pts,
            'mamba_features': mamba_features,}
        mamba_dict_batch.append(mamba_dict)

    save_pkl(mamba_dict_batch, "mamba_dicts_bs%d"%batch_dict['batch_size'])


    
    # # octree_merged, pt_merged = process_voxels(voxels)
    # save_pkl(octree_merged, "octree_merged")
    # save_pkl(pt_merged, "pt_merged")
    
    
    # data = octree_feature(octree_merged)
    # save_pkl(data, "data")

    # logits = ptmamba(data, octree_merged, ptmamba.depth)
    # save_pkl(logits, "logits")

    # mamba_features = []
    # for pts_voxel in torch.unbind(voxels, dim=0):
    #     pts = pts_voxel[:, :3]
    #     octrees = [points2octree(pt) for pt in pts]
    #     octree = ocnn.octree.merge_octrees(octrees)
    #     octree.construct_all_neigh()
    #     pts_searched=ocnn.octree.merge_points(pts)

    print("End Debug")

    # pooled_feature_dim=256
    # roi_ip_feature_dim=128

    # rois = torch.rand((B*N, 7), dtype=torch.float32)
    # pooled_features = torch.rand((B, N, pooled_feature_dim), dtype=torch.float32)
    # proposal_labels = torch.rand((1, B*N), dtype=torch.float32)
    # proposal_boxes_ip = torch.rand((B, N, roi_ip_feature_dim), dtype=torch.float32)

    # batch_dict = {
    #     'rois': rois,  # Random positions for 10 batches of 100 proposals each
    #     'pooled_features': pooled_features,  # Random 16-dimensional features for 10 batches of 100 proposals each
    #     'roi_labels': proposal_labels,  # Random labels for 10 batches of 100 proposals each
    #     'roi_ip_features': proposal_boxes_ip,
    # }