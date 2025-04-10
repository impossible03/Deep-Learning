"""
resnet in pytorch

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

from torchsummary import summary
import torch
import torch.nn as nn
import torchsummary
import matplotlib.pyplot as plt


class BasicBlock(nn.Module):

    # BasicBlock模块的扩张因子，默认为1
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """
            初始化BasicBlock模块。

            参数:
            in_channels (int): 输入特征图的通道数。
            out_channels (int): 输出特征图的通道数。
            stride (int, 可选): 卷积层的步长，默认为1。
        """
        super().__init__()
        # 初始化Shrinkage模块，用于特征收缩
        self.shrinkage = Shrinkage(out_channels, gap_size=(1, 1))

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3),
                      stride=stride, padding=(0, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion,
                      kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            # 应用Shrinkage模块
            self.shrinkage
        )
        # 初始化捷径连接，初始为空序列
        self.shortcut = nn.Sequential()

        # 当步长不为1或者输入通道数与输出通道数不匹配时
        # 使用1x1卷积来调整捷径连接的维度
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                # 1x1卷积层，用于调整通道数
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):

        # 首先将输入 x 传入 self.residual_function 进行处理
        # self.residual_function 是一个 nn.Sequential 对象，包含了多个卷积层、批量归一化层和激活函数
        # 它用于提取输入特征图的深层次特征
        residual_output = self.residual_function(x)

        # 接着将输入 x 传入 self.shortcut 进行处理
        # self.shortcut 是一个捷径连接，用于解决深度神经网络中的梯度消失和梯度爆炸问题
        # 当步长不为 1 或者输入通道数与输出通道数不匹配时，self.shortcut 会包含一个 1x1 卷积层来调整通道数
        shortcut_output = self.shortcut(x)

        # 将 residual_function 和 shortcut 的输出相加
        # 这种相加操作是残差块的核心，它允许网络学习到残差信息，从而更容易训练更深的网络
        combined_output = residual_output + shortcut_output

        # 最后，将相加后的结果通过 ReLU 激活函数进行非线性变换
        # inplace=True 表示直接在原张量上进行修改，这样可以节省内存
        final_output = nn.ReLU(inplace=True)(combined_output)

        return final_output


class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        # 调用父类 nn.Module 的构造函数，确保父类的初始化逻辑被执行
        super(Shrinkage, self).__init__()

        # 初始化自适应平均池化层，将输入特征图的大小调整为 gap_size
        # 自适应平均池化层会根据输入特征图的大小自动计算池化窗口的大小和步长，使得输出特征图的大小为 gap_size
        self.gap = nn.AdaptiveAvgPool2d(gap_size)

        # 初始化一个包含多个层的序列 self.fc
        self.fc = nn.Sequential(
            # 第一个全连接层，将输入通道数为 channel 的特征映射到通道数为 channel 的特征
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            # 第二个全连接层
            nn.Linear(channel, channel),
            # Sigmoid 激活函数，将输出值映射到 [0, 1] 区间，用于生成收缩因子
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 保存输入的原始值，后续用于恢复符号
        x_raw = x
        # 计算输入的绝对值
        x = torch.abs(x)
        # 保存绝对值结果，后续软阈值操作会用到
        x_abs = x
        # 对输入的绝对值进行自适应平均池化操作，将特征图大小调整为指定的 gap_size
        x = self.gap(x)
        # 将池化后的特征图展平为一维向量
        x = torch.flatten(x, 1)
        average = x
        # 将展平后的特征图输入到全连接层序列 self.fc 中进行处理，得到收缩因子
        x = self.fc(x)
        # 调整特征幅度，为软阈值操作做准备
        x = torch.mul(average, x)
        # 在最后两个维度上增加维度，恢复为特征图的形状
        x = x.unsqueeze(2).unsqueeze(2)
        # 软阈值操作：计算绝对值特征图与前面计算结果的差值
        sub = x_abs - x
        # 创建一个全零的张量，用于后续的最大值比较
        zeros = torch.zeros_like(sub)
        # 取差值和零中的最大值，小于零的部分置为零
        n_sub = torch.max(sub, zeros)
        # 恢复原始输入的符号，得到最终的特征收缩结果
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x


class DRSNet(nn.Module):

    def __init__(self, block, num_block, num_classes=4):
        super().__init__()

        self.in_channels = 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=[1, 3], padding=(0, 1), bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 4, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 8, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 16, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 32, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
            make rsnet layers(by layer i didnt mean this 'layer' was the
            same as a neuron netowork layer, ex. conv layer), one layer may
            contain more than one residual shrinkage block

            Args:
                block: block type, basic block or bottle neck block
                out_channels: output depth channel number of this layer
                num_blocks: how many blocks per layer
                stride: the stride of the first block of this layer

            Return:
                return a rsnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def DRSNet18():
    """ return a RsNet 18 object
    """
    return DRSNet(BasicBlock, [2, 2, 2, 2])


def DRSNet34():
    """ return a DRSNet 34 object
    """
    return DRSNet(BasicBlock, [3, 4, 6, 3])


def DRSNet50():
    """ return a ResNet 50 object
    """
    return DRSNet(BottleNeck, [3, 4, 6, 3])


model = DRSNet34()
# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

summary(model, input_size=(1, 2048, 1), batch_size=-1, device='cuda')