import torch
from torchviz import make_dot
from mmdet.models import ResNet
from mmdet.models import FPN
from mmdet.models import RPNHead
from mmcv import ConfigDict


def show_model(self, inputs):
    # 可视化网络
    g = make_dot(self(inputs), params=dict(self.named_parameters()))
    g.view()


if __name__ == '__main__':
    backbone_cfgs = {'depth': 50, 'num_stages': 4, 'out_indices': (0, 1, 2, 3), 'frozen_stages': -1,
                     'norm_cfg': {'type': 'BN', 'requires_grad': True}, 'norm_eval': True, 'style': 'pytorch'}
    # 初始化骨架网络
    backbone = ResNet(**backbone_cfgs)
    inputs = torch.rand(1, 3, 128, 128)
    backbone_level_outputs = backbone.forward(inputs)
    print("=" * 100)
    print("第一步载入骨架网络...")
    for level_out in backbone_level_outputs:
        print("输出骨架网络提取的特征图:", tuple(level_out.shape))
    # 验证输出四个特征图深度分别为(256,512,1024,2048)
    # show_model(self, inputs)
    print("=" * 100)
    # 网络颈部是FPN网络
    # 初始化颈部网络
    # 对应的ResNet输出每个步骤的特征图将之称为C2,C3,C4,C5
    print("第二步载入颈部网络...")
    # 这里输出的num_outs为5,resnet提取的特征图为4,所以最后FPN会用最大池化操作下采样P5多加一层
    neck_cfgs = {'in_channels': [256, 512, 1024, 2048], 'out_channels': 256, 'num_outs': 5}
    neck = FPN(**neck_cfgs)
    neck_level_outputs = neck.forward(backbone_level_outputs)
    for level_out in neck_level_outputs:
        print("FPN输出:", tuple(level_out.shape))
    print("=" * 100)
    # show_model(neck, backbone_level_outputs)
    print("第三步载入头部网络...")
    head_cfgs = {
        'in_channels': 256,
        'feat_channels': 256,
        'anchor_generator': {
            'type': 'AnchorGenerator',
            'scales': [8],
            'ratios': [0.5, 1.0, 2.0],
            'strides': [4, 8, 16, 32, 64]
        },
        'bbox_coder': {
            'type': 'DeltaXYWHBBoxCoder',
            'target_means': [0.0, 0.0, 0.0, 0.0],
            'target_stds': [1.0, 1.0, 1.0, 1.0]
        },
        'loss_cls': {
            'type': 'CrossEntropyLoss',
            'use_sigmoid': True,
            'loss_weight': 1.0
        },
        'loss_bbox': {
            'type': 'L1Loss',
            'loss_weight': 1.0
        },
        'train_cfg': {
            'assigner': {
                'type': 'MaxIoUAssigner',
                'pos_iou_thr': 0.7,
                'neg_iou_thr': 0.3,
                'min_pos_iou': 0.3,
                'match_low_quality': True,
                'ignore_iof_thr': -1
            },
            'sampler': {
                'type': 'RandomSampler',
                'num': 256,
                'pos_fraction': 0.5,
                'neg_pos_ub': -1,
                'add_gt_as_proposals': False
            },
            'allowed_border': -1,
            'pos_weight': -1,
            'debug': False
        },
        'test_cfg': {
            'nms_across_levels': False,
            'nms_pre': 1000,
            'nms_post': 1000,
            'max_num': 1000,
            'nms_thr': 0.7,
            'min_bbox_size': 0
        }}
    head_cfgs = ConfigDict(head_cfgs)
    rpn_head = RPNHead(**head_cfgs)

