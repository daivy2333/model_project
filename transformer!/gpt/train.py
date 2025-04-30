import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
from utils import build_vocab, encode, decode
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 配置
block_size = 128
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.1
lr = 3e-4
batch_size = 64
max_steps = 1500

# 数据准备
text = Path("transformer!\\gpt\\input.txt").read_text(encoding='utf-8')
vocab_size = build_vocab(text)
data = torch.tensor(encode(text), dtype=torch.long)
train_data, val_data = data[:int(0.9*len(data))], data[int(0.9*len(data)):]

# 模型
config = GPTConfig(vocab_size, block_size, n_layer, n_head, n_embd, dropout)
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# 批次函数
def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# 训练
for step in range(max_steps):
    x, y = get_batch(train_data)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# 保存模型
torch.save(model.state_dict(), "gpt_model.pt")
print("✅ 模型已保存为 gpt_model.pt")