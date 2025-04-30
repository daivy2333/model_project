import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_dim=768, img_size=224):
        # 初始化父类
        super().__init__()
        # 设置补丁大小
        self.patch_size = patch_size
        # 计算图像中补丁的数量，假设图像是正方形的
        self.n_patches = (img_size // patch_size) ** 2
        # 定义一个二维卷积层，用于将图像分割成补丁并投影到指定的嵌入维度
        # in_channels: 输入通道数，默认为3（RGB图像）
        # emb_dim: 输出通道数，即嵌入维度
        # kernel_size: 卷积核大小，等于补丁大小
        # stride: 步长，等于补丁大小，确保补丁不重叠
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2)  # (B, D, N)
        x = x.transpose(1, 2)  # (B, N, D)
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=10,
                 emb_dim=768, depth=6, heads=8, mlp_dim=2048, dropout=0.1):
        # 初始化ViT模型，设置图像大小、补丁大小、输入通道数、分类数、嵌入维度、编码器层数、注意力头数、MLP维度和dropout率
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_dim, img_size)
        # 创建补丁嵌入层，将图像分割成补丁并转换为嵌入向量
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        # 创建[CLS] token，用于表示整个图像的嵌入
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, emb_dim))
        # 创建位置嵌入，用于编码补丁的位置信息
        self.dropout = nn.Dropout(dropout)

        # 创建dropout层，用于防止过拟合
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=heads,
                                                   dim_feedforward=mlp_dim, dropout=dropout, activation='gelu')
        # 创建Transformer编码器层，设置嵌入维度、注意力头数、前馈网络维度、dropout率和激活函数
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 创建Transformer编码器，由多个编码器层组成
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

        # 创建MLP头，包括一层LayerNorm和一层全连接层，用于分类任务
    def forward(self, x):
        B = x.size(0)
        # 获取批量大小
        x = self.patch_embed(x)  # (B, N, D)

        # 将输入图像转换为补丁嵌入向量
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        # 将[CLS] token扩展到批量大小
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, D)
        # 将[CLS] token与补丁嵌入向量拼接
        x = x + self.pos_embed
        # 添加位置嵌入
        x = self.dropout(x)

        # 应用dropout
        x = self.transformer(x)  # (B, N+1, D)
        # 通过Transformer编码器处理嵌入向量
        cls_output = x[:, 0]  # 取 [CLS] token
        # 提取[CLS] token的输出
        out = self.mlp_head(cls_output)
        # 通过MLP头进行分类
        return out

# 测试代码：
if __name__ == "__main__":
    model = ViT()
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    print(out.shape)  # 应该是 [2, 10]
