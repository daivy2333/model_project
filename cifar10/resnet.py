import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride= 1):
        super(BasicBlock, self).__init__()
        # 第一个卷积层，输入通道数为in_channels，输出通道数为out_channels，卷积核大小为3，步长为stride，填充为1
        self.conv1= nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # 第一个批归一化层，输入通道数为out_channels
        self.bn1= nn.BatchNorm2d(out_channels)
        # ReLU激活函数，inplace=True表示在原地进行操作，节省内存
        self.relu= nn.ReLU(inplace=True)

        # 第二个卷积层，输入通道数为out_channels，输出通道数为out_channels，卷积核大小为3，步长为1，填充为1
        self.conv2= nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 第二个批归一化层，输入通道数为out_channels
        self.bn2= nn.BatchNorm2d(out_channels)

        # 如果输入和输出的通道数不同，则需要进行线性变换
        self.downsample= None
        if stride!= 1 or in_channels!= out_channels:
            self.downsample= nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity= x # 保存输入

        out= self.conv1(x)
        out= self.bn1(out)
        out= self.relu(out)

        out= self.conv2(out)
        out= self.bn2(out)

        if self.downsample is not None:
            identity= self.downsample(x)

        out+= identity # 残差连接
        out= self.relu(out)

        return out
    

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes= 10):
        # 初始化ResNet类的构造函数，参数包括block（残差块类型）、layers（每层中残差块的数量列表）、num_classes（输出类别数，默认为10）
        super(ResNet, self).__init__()
        self.in_channels= 64 # 输入通道数

        self.conv1= nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1) # 输入通道数为3，输出通道数为64，卷积核大小为3，步长为1，填充为1
        self.bn1= nn.BatchNorm2d(64) # 批归一化层，输入通道数为64
        self.relu= nn.ReLU(inplace=True) # ReLU激活函数，inplace=True表示在原地进行操作，节省内存

        self.layer1= self.make_layer(block, 64, layers[0], stride=1) # 构建第一层，输出通道数为64，包含layers[0]个残差块，步长为1
        self.layer2= self.make_layer(block, 128, layers[1], stride=2) # 构建第二层，输出通道数为128，包含layers[1]个残差块，步长为2
        self.layer3= self.make_layer(block, 256, layers[2], stride=2) # 构建第三层，输出通道数为256，包含layers[2]个残差块，步长为2
        self.layer4= self.make_layer(block, 512, layers[3], stride=2) # 构建第四层，输出通道数为512，包含layers[3]个残差块，步长为2

        self.avg_pool= nn.AdaptiveAvgPool2d((1, 1)) # 平均池化层，输出大小为1x1

        self.fc= nn.Linear(512, num_classes) # 全连接层，输入特征数为512，输出类别数为num_classes

    def make_layer(self, block, out_channels, blocks, stride):

        # 构建一个包含多个残差块的层
        strides= [stride] + [1]*(blocks-1) # 如果有多个残差块，则第一个残差块的步长为stride，其余的步长为1

        layers= []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride)) # 添加一个残差块
            self.in_channels= out_channels # 更新输入通道数

        return nn.Sequential(*layers) # 将所有残差块合并为一个序列，然后返回

    def forward(self, x):

        # 前向传播函数
        x= self.conv1(x) # 卷积操作
        x= self.bn1(x) # 批归一化
        x= self.relu(x) # ReLU激活

        x= self.layer1(x) # 通过第一层残差块
        x= self.layer2(x) # 通过第二层残差块
        x= self.layer3(x) # 通过第三层残差块
        x= self.layer4(x) # 通过第四层残差块

        x= self.avg_pool(x) # 平均池化
        x= torch.flatten(x, 1) # 将张量展平，从第1维开始
        x= self.fc(x) # 全连接层

        return x # 返回输出
    


model= ResNet(BasicBlock, [2, 2, 2, 2], num_classes= 10) # 创建ResNet模型，使用BasicBlock作为残差块，每层的残差块数量为[2, 2, 2, 2]

print(model) # 打印模型结构