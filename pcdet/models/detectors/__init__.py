from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .PartA2_mamba import PartA2Net_Mamba
from .PartA2_relation_mamba import PartA2NetRelationMamba
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pointpillar_fps import PointPillarFPS
from .pv_rcnn import PVRCNN
from .pv_rcnn_mamba import PVRCNN_Mamba
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .voxel_rcnn_mamba import VoxelRCNN_Mamba
from .centerpoint import CenterPoint
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .mppnet import MPPNet
from .mppnet_e2e import MPPNetE2E
from .pillarnet import PillarNet
from .voxelnext import VoxelNeXt
from .transfusion import TransFusion
from .bevfusion import BevFusion
from .pv_rcnn_relation import PVRCNNRelation
from .pv_rcnn_relation_mamba import PVRCNNRelation_Mamba
from .pv_rcnn_plusplus_relation import PVRCNNPlusPlusRelation
from .centerpoint_twostage import CenterPointTwoStage
from .PartA2_relation_net import PartA2NetRelation
from .voxel_rcnn_relation import VoxelRCNNRelation

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PartA2Net_Mamba': PartA2Net_Mamba,
    'PVRCNN': PVRCNN,
    'PVRCNN_Mamba': PVRCNN_Mamba,
    'PointPillar': PointPillar,
    'PointPillarFPS': PointPillarFPS,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint,
    'PillarNet': PillarNet,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'MPPNet': MPPNet,
    'MPPNetE2E': MPPNetE2E,
    'PillarNet': PillarNet,
    'VoxelNeXt': VoxelNeXt,
    'TransFusion': TransFusion,
    'BevFusion': BevFusion,
    'PVRCNNRelation': PVRCNNRelation,
    'PVRCNNRelation_Mamba': PVRCNNRelation_Mamba,
    'PVRCNNPlusPlusRelation': PVRCNNPlusPlusRelation,
    'CenterPointTwoStage': CenterPointTwoStage,
    'PartA2NetRelation': PartA2NetRelation,
    'PartA2NetRelationMamba': PartA2NetRelationMamba,
    'VoxelRCNNRelation': VoxelRCNNRelation,
    'VoxelRCNN_Mamba': VoxelRCNN_Mamba,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
