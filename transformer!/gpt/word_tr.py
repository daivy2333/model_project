import torch
import pickle
from pathlib import Path

# 1. 加载文本（按单词处理）
raw_text = Path("transformer!/gpt/input.txt").read_text(encoding='utf-8')

# 2. 添加特殊 token：<s> 开头，<eos> 结尾
# 将每一行处理为句子（可根据需要替换为其他分割方法）
lines = raw_text.strip().splitlines()
tokenized_lines = [["<s>"] + line.strip().split() + ["<eos>"] for line in lines]
words = [word for line in tokenized_lines for word in line]

# 3. 构建词表
vocab = sorted(set(words))
vocab.append("<unk>")
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}

# 4. 保存词表
with open("transformer!/gpt/vocab.pkl", "wb") as f:
    pickle.dump((stoi, itos), f)

# 5. 编码文本为 token ID
def encode(word_list):
    return [stoi.get(w, stoi["<unk>"]) for w in word_list]

data_ids = torch.tensor(encode(words), dtype=torch.long)

# 6. 划分训练/验证集
n = int(0.9 * len(data_ids))
train_data = data_ids[:n]
val_data = data_ids[n:]

# 7. 保存编码后的数据
Path("transformer!/gpt/data").mkdir(parents=True, exist_ok=True)
torch.save(train_data, "transformer!/gpt/data/train.pt")
torch.save(val_data, "transformer!/gpt/data/val.pt")

print("✅ 词级数据和词表已准备完成")
