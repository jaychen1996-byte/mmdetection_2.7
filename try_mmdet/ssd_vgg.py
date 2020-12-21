import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import VGG


class SSDVGG(VGG):
    """VGG Backbone network for single-shot-detection.

    Args:
        input_size (int): width and height of input, from {300, 512}.
        depth (int): Depth of vgg, from {11, 13, 16, 19}.
        out_indices (Sequence[int]): Output from which stages.

    Example:
        >>> self = SSDVGG(input_size=300, depth=11)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 300, 300)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 1024, 19, 19)
        (1, 512, 10, 10)
        (1, 256, 5, 5)
        (1, 256, 3, 3)
        (1, 256, 1, 1)
    """
    # 对应输入size为300和512时输出层的深度
    extra_setting = {
        300: (256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256),
        512: (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128),
    }

    def __init__(self,
                 # width and height of input, from {300, 512}.
                 input_size,
                 # Depth of vgg, from {11, 13, 16, 19}.
                 depth,
                 # 是否需要末层池化
                 with_last_pool=False,
                 # when True, will use `ceil` instead of `floor` to compute the output shape（Maxpooling2D）
                 ceil_mode=True,
                 # Output from which stages.
                 out_indices=(3, 4),
                 # 提取的特征层列表下标
                 out_feature_indices=(22, 34),
                 l2_norm_scale=20.):
        # TODO: in_channels for mmcv.VGG
        # 继承自mmcv.VGG模块
        super(SSDVGG, self).__init__(
            # VGG网络的深度,可选值有{11, 13, 16, 19}.
            depth,
            # 是否需要末层池化
            with_last_pool=with_last_pool,
            # 设置为Ture时,在Maxpooling2D层中,使用"ceil"模式计算输出Shape,反之使用"floor"模式
            ceil_mode=ceil_mode,
            # Output from which stages.
            out_indices=out_indices)
        # 判断输入是否是300和512其一
        assert input_size in (300, 512)
        # 赋值给self.input_size属性
        self.input_size = input_size
        # 继承自VGG的self.features属性
        # 添加MaxPool2d层
        self.features.add_module(
            str(len(self.features)),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        # 添加Conv2d层 kernel_size=3, padding=6, dilation=6
        self.features.add_module(
            str(len(self.features)),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6))
        # 添加激活函数Relu
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))
        # 添加Conv2d层 kernel_size=1
        self.features.add_module(
            str(len(self.features)), nn.Conv2d(1024, 1024, kernel_size=1))
        # 添加激活函数Relu
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))
        # 更新赋值out_feature_indices到self.out_feature_indices
        self.out_feature_indices = out_feature_indices
        # 输入深度,创建SSD基于VGG的extra_layers
        self.inplanes = 1024
        self.extra = self._make_extra_layers(self.extra_setting[input_size])
        self.l2_norm = L2Norm(  # 初始化L2正则化函数
            self.features[out_feature_indices[0] - 1].out_channels,
            l2_norm_scale)

    def forward(self, x):
        """Forward function."""
        # 前向推理输出
        outs = []
        for i, layer in enumerate(self.features):  # 在设置with_last_pool=False的情况下,self.features在VGG基础架构上添加了5层
            x = layer(x)  # 取出每个网络层
            if i in self.out_feature_indices:  # 是否在提取特征层列表里
                outs.append(x)  # 若是,添加到输出列表中
        for i, layer in enumerate(self.extra):  # 迭代extra_layers
            x = F.relu(layer(x), inplace=True)  # 默认卷积层后添加Relu激活函数
            if i % 2 == 1:  # 如SSD_VGG网络架构一样,在进行1×1×channels卷积后提取特征,添加到输出列表中
                outs.append(x)
        outs[0] = self.l2_norm(outs[0])  # L2正则化
        # 返回结果
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    # 创建extra_layers
    def _make_extra_layers(self, outplanes):
        layers = []  # 层列表
        kernel_sizes = (1, 3)  # 根据SSD_VGG网络结构可得扩展层内的内核大小非1即3
        num_layers = 0  # 扩展层内计数
        outplane = None  # 初始化输出层深度变量,首先第一个输入的self.inplanes是1024
        # 根据预先设置好的extra_setting取值
        for i in range(len(outplanes)):
            # 在extra_setting中插入'S'区分是否进行步长为2的卷积
            if self.inplanes == 'S':
                # 更新输入的通道
                self.inplanes = outplane
                continue
            # 获取到kernel值
            k = kernel_sizes[num_layers % 2]
            # 此处判断是否进行步长为2的卷积
            if outplanes[i] == 'S':
                outplane = outplanes[i + 1]
                conv = nn.Conv2d(
                    self.inplanes, outplane, k, stride=2, padding=1)
            else:
                outplane = outplanes[i]
                conv = nn.Conv2d(
                    self.inplanes, outplane, k, stride=1, padding=0)
            # 添加到层列表中
            layers.append(conv)
            self.inplanes = outplanes[i]
            num_layers += 1
        # 如果输入size是512添加一个层
        if self.input_size == 512:
            layers.append(nn.Conv2d(self.inplanes, 256, 4, padding=1))

        return nn.Sequential(*layers)


class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20., eps=1e-10):
        """L2 normalization layer.

        Args:
            n_dims (int): Number of dimensions to be normalized
            scale (float, optional): Defaults to 20..
            eps (float, optional): Used to avoid division by zero.
                Defaults to 1e-10.
        """
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        """Forward function."""
        # normalization layer convert to FP32 in FP16 training
        x_float = x.float()
        norm = x_float.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return (self.weight[None, :, None, None].float().expand_as(x_float) *
                x_float / norm).type_as(x)


if __name__ == '__main__':
    # 实例化SSD_VGG网络
    self = SSDVGG(input_size=300, depth=16)
    # 转为评估模式
    self.eval()
    # 随机输入
    inputs = torch.rand(1, 3, 300, 300)
    # 获取输出
    level_outputs = self.forward(inputs)
    # 迭代输出
    for level_out in level_outputs:
        print(tuple(level_out.shape))

    """
    Output:
        (1, 1024, 19, 19)
        (1, 512, 10, 10)
        (1, 256, 5, 5)
        (1, 256, 3, 3)
        (1, 256, 1, 1)
    """
