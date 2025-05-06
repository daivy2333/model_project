# train.py
import torch
import torch.nn.functional as F
import pickle
from model import GPT, GPTConfig
from pathlib import Path
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# 1. 环境配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter(log_dir="transformer!/gpt/runs")

# 2. 加载词表
with open("transformer!/gpt/vocab.pkl", "rb") as f:
    stoi, itos = pickle.load(f)
vocab_size = len(stoi)

# 3. 超参数配置
block_size = 64  # 句子长度（单词数）
n_layer = 8
n_head = 8
n_embd = 256
dropout = 0.1
lr = 3e-4
batch_size = 32
max_steps = 50000

# 4. 加载预处理数据
train_data = torch.load("transformer!/gpt/data/train.pt")
val_data = torch.load("transformer!/gpt/data/val.pt")

# 5. 构建模型
config = GPTConfig(vocab_size, block_size, n_layer, n_head, n_embd, dropout)
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = GradScaler()

# 6. 生成训练批次
def get_batch(data):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# 7. 训练循环
for step in range(max_steps):
    x, y = get_batch(train_data)

    # AMP 混合精度训练
    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    writer.add_scalar("Loss/Train", loss.item(), step)

    # 验证与打印
    if step % 100 == 0:
        model.eval()
        with torch.no_grad(), autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            x_val, y_val = get_batch(val_data)
            val_logits = model(x_val)
            val_loss = F.cross_entropy(val_logits.view(-1, vocab_size), y_val.view(-1))
        writer.add_scalar("Loss/Val", val_loss.item(), step)
        print(f"Step {step}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        torch.cuda.empty_cache()
        model.train()

# 8. 保存模型
torch.save(model.state_dict(), "transformer!/gpt/gpt_model.pt")
print("✅ 训练完成，模型已保存")
writer.close()
