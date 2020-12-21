# Copyright (c) Open-MMLab. All rights reserved.
import logging

import torch.nn as nn
import torch


# 3*3卷积层
def conv3x3(in_planes, out_planes, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        padding=dilation,
        dilation=dilation)


# 生成VGG层
def make_vgg_layer(inplanes,
                   planes,
                   num_blocks,
                   dilation=1,
                   with_bn=False,
                   ceil_mode=False):
    # 创建空列表
    layers = []
    # 根据每步中的块数创建网络层
    for _ in range(num_blocks):
        # 添加卷积层
        layers.append(conv3x3(inplanes, planes, dilation))
        # 如果添加BN层则添加
        if with_bn:
            layers.append(nn.BatchNorm2d(planes))
        # 添加激活函数Relu
        layers.append(nn.ReLU(inplace=True))
        # 将下层输出深度赋值给上层
        inplanes = planes
    # 创建完CONV、BN层后添加一最大池化层（这是）num_modules最后+1的原因
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))

    return layers


class VGG(nn.Module):
    """VGG backbone.

    Args:
        depth (int): Depth of vgg, from {11, 13, 16, 19}.
        with_bn (bool): Use BatchNorm or not.
        num_classes (int): number of classes for classification.
        num_stages (int): VGG stages, normally 5.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers as eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
    """

    # 每阶段的块数
    arch_settings = {
        11: (1, 1, 2, 2, 2),
        13: (2, 2, 2, 2, 2),
        16: (2, 2, 3, 3, 3),
        19: (2, 2, 4, 4, 4)
    }

    def __init__(self,
                 depth,
                 with_bn=False,
                 num_classes=-1,
                 num_stages=5,
                 dilations=(1, 1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3, 4),
                 frozen_stages=-1,
                 bn_eval=True,
                 bn_frozen=False,
                 ceil_mode=False,  # when True, will use `ceil` instead of `floor` to compute the output shape
                 with_last_pool=True):
        super(VGG, self).__init__()  # 继承自nn.Module模块
        # 判断是否是 Depth of vgg, from {11, 13, 16, 19}.
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for vgg')
        # VGG网络 VGG stages, normally 5
        assert num_stages >= 1 and num_stages <= 5
        # 得到每个阶段的块数
        stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        # 判断长度与输出层的最大下标
        assert len(dilations) == num_stages
        assert max(out_indices) <= num_stages

        self.num_classes = num_classes  # number of classes for classification.
        self.out_indices = out_indices  # Output from which stages.
        self.frozen_stages = frozen_stages  # Stages to be frozen (all param fixed). -1 means not freezing any parameters.
        self.bn_eval = bn_eval  # Whether to set BN layers as eval mode, namely, freeze running stats (mean and var).
        self.bn_frozen = bn_frozen  # Whether to freeze weight and bias of BN layers.

        self.inplanes = 3  # 输入数据的深度，例如（8,3,256,256），其中inplanes=3
        start_idx = 0  # 始点
        vgg_layers = []  # VGG层列表
        self.range_sub_modules = []  # 记录每步在vgg_layers列表中的下标位置
        # 迭代出每步中的块数
        for i, num_blocks in enumerate(self.stage_blocks):
            # 每个小块里如果包含bn层即为3层,最后一层
            num_modules = num_blocks * (2 + with_bn) + 1
            # 终点
            end_idx = start_idx + num_modules
            # 是否空洞卷积
            dilation = dilations[i]
            # 输出层深度
            planes = 64 * 2 ** i if i < 4 else 512
            # 添加VGG层
            vgg_layer = make_vgg_layer(
                self.inplanes,
                planes,
                num_blocks,
                dilation=dilation,
                with_bn=with_bn,
                ceil_mode=ceil_mode)
            # 将创建的VGG层添加到vgg_layers列表
            vgg_layers.extend(vgg_layer)
            # 将输出层深度赋值给输入层
            self.inplanes = planes
            self.range_sub_modules.append([start_idx, end_idx])
            start_idx = end_idx
        # 判断是否需要末层池化
        if not with_last_pool:
            vgg_layers.pop(-1)
            self.range_sub_modules[-1][1] -= 1
        # 命名模块
        self.module_name = 'features'
        # 将子模块添加到当前模块，将list转为torch模块
        self.add_module(self.module_name, nn.Sequential(*vgg_layers))

        # 如果num_classes大于0则添加最后的全连接层
        if self.num_classes > 0:
            # A sequential container named classifier.
            self.classifier = nn.Sequential(
                # Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
                nn.Linear(512 * 7 * 7, 4096),
                # 激活函数
                nn.ReLU(True),
                # Dropout
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

    # 前向推理
    def forward(self, x):
        # 输出列表
        outs = []
        # getattr() 函数用于返回一个对象属性值。
        vgg_layers = getattr(self, self.module_name)

        # 嵌套循环
        # a = [[0, 3], [3, 6], [6, 11], [11, 16], [16, 21]]
        # b = {"a": 1, "b": 2, "c": 3}
        # 获取到stage_blocks的长度等于网络的stages数
        for i in range(len(self.stage_blocks)):
            # range_sub_modules是一个二维的列表在列表前添加*号会把列表中的属性迭代出来,相当于"for m in range_sub_modules"
            # 当在字典前添加*号时会把字典的每个key输出
            # 在这里例如[[0, 3], [3, 6], [6, 11], [11, 16], [16, 21]],会输出range(0,3)
            for j in range(*self.range_sub_modules[i]):
                # 根据下标取到网络层
                vgg_layer = vgg_layers[j]
                # 将输入向前推理并更新赋值
                x = vgg_layer(x)
            # 判断是否提取特征的Stage,是的话存入outs列表中
            if i in self.out_indices:
                outs.append(x)
        # 如果num_classes大于0,传入分类器（全连接层）进行分类
        if self.num_classes > 0:
            # 向量铺平
            x = x.view(x.size(0), -1)
            # 传入分类器
            x = self.classifier(x)
            outs.append(x)
        # 结果返回
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)


if __name__ == '__main__':
    # 实例化模型
    mdoel = VGG(11, num_classes=2)
    # 随机变量维度是（1,3,300,300）,batch,inplanes,img_w,img_h
    inputs = torch.rand(1, 3, 224, 224)
    # 进行前向推理
    level_outputs = mdoel(inputs)
    # 输出每层维度
    for level_out in level_outputs:
        print(tuple(level_out.shape))
