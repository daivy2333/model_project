import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTConfig:
    # 初始化方法，用于创建GPTConfig类的实例
    def __init__(self, vocab_size, block_size=128, n_layer=4, n_head=4, n_embd=256, dropout=0.1):
        # vocab_size: 词汇表大小，表示模型可以处理的唯一标记的数量
        self.vocab_size = vocab_size
        # block_size: 上下文窗口大小，表示模型在一次前向传播中可以处理的序列长度
        self.block_size = block_size
        # n_layer: Transformer层的数量，表示模型的深度
        self.n_layer = n_layer
        # n_head: 多头注意力机制中的注意力头数，表示每个注意力头负责的部分注意力计算
        self.n_head = n_head
        # n_embd: 嵌入维度，表示每个标记的向量表示的维度
        self.n_embd = n_embd
        # dropout: Dropout率，用于防止过拟合，表示在训练过程中随机丢弃的比例
        self.dropout = dropout

class SelfAttention(nn.Module):
    def __init__(self, config):
        # 初始化父类
        super().__init__()
        # 定义线性变换层，用于计算key
        self.key = nn.Linear(config.n_embd, config.n_embd)
        # 定义线性变换层，用于计算query
        self.query = nn.Linear(config.n_embd, config.n_embd)
        # 定义线性变换层，用于计算value
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # 定义线性变换层，用于输出投影
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(config.dropout)
        # 定义注意力头的数量
        self.n_head = config.n_head

    def forward(self, x):
        # 获取输入的批次大小、序列长度和嵌入维度
        B, T, C = x.size()
        # 计算每个注意力头的维度
        head_dim = C // self.n_head
        # 计算key，并调整形状以适应多头注意力
        k = self.key(x).view(B, T, self.n_head, head_dim).transpose(1, 2)
        # 计算query，并调整形状以适应多头注意力
        q = self.query(x).view(B, T, self.n_head, head_dim).transpose(1, 2)
        # 计算value，并调整形状以适应多头注意力
        v = self.value(x).view(B, T, self.n_head, head_dim).transpose(1, 2)

        # 计算注意力分数
        att = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
        # 创建掩码矩阵，用于防止未来信息的泄露
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        # 将注意力分数中不需要的部分设为负无穷
        att = att.masked_fill(mask == 0, float('-inf'))
        # 应用softmax函数，得到注意力权重
        att = F.softmax(att, dim=-1)
        # 应用dropout
        att = self.dropout(att)
        # 计算输出
        out = att @ v
        # 调整输出形状
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        # 应用输出投影
        return self.proj(out)

class TransformerBlock(nn.Module):
    # 定义TransformerBlock类，继承自nn.Module
    def __init__(self, config):
        # 初始化函数，接收配置参数config
        super().__init__()
        # 调用父类nn.Module的初始化函数
        self.sa = SelfAttention(config)
        # 初始化自注意力机制模块SelfAttention，传入配置参数config
        self.ln1 = nn.LayerNorm(config.n_embd)
        # 初始化第一个层归一化模块LayerNorm，输入维度为config.n_embd
        self.ff = nn.Sequential(
            # 初始化前馈神经网络模块，包含以下层：
            nn.Linear(config.n_embd, 4 * config.n_embd),
            # 线性层，将输入维度从config.n_embd变换到4倍的config.n_embd
            nn.ReLU(),
            # ReLU激活函数
            nn.Linear(4 * config.n_embd, config.n_embd),
            # 线性层，将输入维度从4倍的config.n_embd变换回config.n_embd
            nn.Dropout(config.dropout),
            # Dropout层，防止过拟合，丢弃率为config.dropout
        )
        self.ln2 = nn.LayerNorm(config.n_embd)

        # 初始化第二个层归一化模块LayerNorm，输入维度为config.n_embd
    def forward(self, x):
        # 定义前向传播函数，接收输入x
        x = x + self.sa(self.ln1(x))
        # 计算自注意力机制输出，并进行残差连接，输入x经过第一个层归一化后传入自注意力模块
        x = x + self.ff(self.ln2(x))
        # 计算前馈神经网络输出，并进行残差连接，输入x经过第二个层归一化后传入前馈神经网络模块
        return x

class GPT(nn.Module):
    def __init__(self, config):
        # 初始化父类nn.Module
        super().__init__()
        # 创建词嵌入层，将词汇表中的每个词映射到n_embd维的向量
        self.token_embed = nn.Embedding(config.vocab_size, config.n_embd)
        # 创建位置嵌入层，将序列中的每个位置映射到n_embd维的向量
        self.pos_embed = nn.Embedding(config.block_size, config.n_embd)
        # 创建多个TransformerBlock，并使用nn.Sequential将它们串联起来
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        # 创建层归一化层，用于对最后的输出进行归一化处理
        self.ln_f = nn.LayerNorm(config.n_embd)
        # 创建线性层，将n_embd维的向量映射到词汇表大小的向量，用于生成预测
        self.head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx):
        # 获取输入的批次大小B和序列长度T
        B, T = idx.size()
        # 创建一个从0到T-1的序列，表示位置信息，并将其移动到与输入相同的设备上
        pos = torch.arange(T, device=idx.device)
        # 将输入的索引通过词嵌入层和位置嵌入层，得到嵌入向量，并进行相加
        x = self.token_embed(idx) + self.pos_embed(pos)
        # 将嵌入向量通过多个TransformerBlock进行前向传播
        x = self.blocks(x)
        # 对最后的输出进行层归一化处理
        x = self.ln_f(x)
        # 将归一化后的输出通过线性层，得到最终的预测结果
        return self.head(x)