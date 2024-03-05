import torch
import torch.nn as nn

class SiLU(nn.Module):
    """SiLU激活函数"""
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
def get_activation(name="silu", inplace=True):
    # inplace为True，将会改变输入的数据 (降低显存)，否则不会改变原输入，只会产生新的输出
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """带归一化和激活函数的标准卷积并且保证宽高不变"""
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        """
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param ksize: 卷积核大小
        :param stride: 步长
        :param groups: 是否分组卷积
        :param bias: 偏置
        :param act: 所选激活函数
        """
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))