import torch.nn as nn
class DoubleConv(nn.Module):
    # 定义一个名为DoubleConv的类，继承自nn.Module，用于实现两层卷积操作
    def __init__(self, in_channels, out_channels):
        # 构造函数，初始化DoubleConv类的实例
        super().__init__()
        # 调用父类nn.Module的构造函数，确保正确初始化
        self.double_conv = nn.Sequential(
            # 定义一个名为double_conv的顺序容器，包含一系列的卷积、批归一化和激活操作
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # 第一层卷积操作，输入通道数为in_channels，输出通道数为out_channels，卷积核大小为3x3，padding为1
            nn.BatchNorm2d(out_channels),
            # 批归一化操作，对第一层卷积的输出进行归一化处理
            nn.ReLU(inplace=True),
            # ReLU激活函数，对归一化后的输出进行非线性变换，inplace=True表示在原地进行操作，节省内存
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # 第二层卷积操作，输入和输出通道数均为out_channels，卷积核大小为3x3，padding为1
            nn.BatchNorm2d(out_channels),
            # 批归一化操作，对第二层卷积的输出进行归一化处理
            nn.ReLU(inplace=True)
            # ReLU激活函数，对归一化后的输出进行非线性变换，inplace=True表示在原地进行操作，节省内存
        )

    def forward(self, x):
        # 定义前向传播函数，输入为x
        return self.double_conv(x)
class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        # 初始化Down类，继承自nn.Module
        super().__init__()
        # 调用父类nn.Module的初始化方法
        self.maxpool_conv = nn.Sequential(
            # 定义一个Sequential容器，包含多个子模块，按顺序执行
            nn.MaxPool2d(2),
            # 最大池化操作，池化窗口大小为2x2，步幅默认为2
            DoubleConv(in_channels, out_channels)
            # DoubleConv是一个自定义的双卷积层模块，用于特征提取
        )

    def forward(self, x):
        # 定义前向传播方法
        return self.maxpool_conv(x)
import torch
import torch.nn.functional as F

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # 初始化Up类，继承自nn.Module
        # in_channels: 输入通道数
        # out_channels: 输出通道数
        # bilinear: 是否使用双线性上采样，默认为True
        if bilinear:
            # 如果使用双线性上采样
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # nn.Upsample: 上采样层，scale_factor=2表示放大两倍，mode='bilinear'表示使用双线性插值，align_corners=True表示对齐角点
        else:
            # 如果不使用双线性上采样，使用转置卷积
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # nn.ConvTranspose2d: 转置卷积层，in_channels为输入通道数，in_channels // 2为输出通道数，kernel_size=2, stride=2表示卷积核大小为2x2，步长为2
        self.conv = DoubleConv(in_channels, out_channels)

        # DoubleConv: 自定义的双卷积层，in_channels为输入通道数，out_channels为输出通道数
    def forward(self, x1, x2):
        # forward方法，定义前向传播过程
        # x1: 输入特征图1
        # x2: 输入特征图2
        x1 = self.up(x1)
        # 对x1进行上采样
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # 计算x2和上采样后的x1在高度和宽度上的差异
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # 使用F.pad对x1进行填充，使其尺寸与x2匹配
        # 填充顺序为：[左, 右, 上, 下]
        x = torch.cat([x2, x1], dim=1)
        # 将x2和填充后的x1在通道维度上进行拼接
        return self.conv(x)
class OutConv(nn.Module):  # 定义一个名为OutConv的类，继承自nn.Module，用于输出卷积层
    def __init__(self, in_channels, out_channels):  # 构造函数，初始化输出卷积层
        super(OutConv, self).__init__()  # 调用父类nn.Module的构造函数
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 初始化一个二维卷积层，卷积核大小为1

    def forward(self, x):  # 定义前向传播函数
        return self.conv(x)  # 对输入x进行卷积操作，并返回结果