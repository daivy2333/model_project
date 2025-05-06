import torch
import torch.nn.functional as F
import pickle
from model import GPT, GPTConfig

# 加载词表
with open("transformer!/gpt/vocab.pkl", "rb") as f:
    stoi, itos = pickle.load(f)

def encode(s):
    return [stoi.get(w, stoi.get('<unk>', 0)) for w in s.split()]

def decode(l):
    words = [itos[i] for i in l]
    return ' '.join(w for w in words if w not in {'<unk>', '<eos>', '<s>'})


def load_model(model_path, device, block_size=64):
    vocab_size = len(stoi)
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=8,
        n_head=8,
        n_embd=256,
        dropout=0.1
    )
    model = GPT(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, config

@torch.no_grad()
def generate(model, config, prompt, device, max_new_tokens=80, temperature=0.7, top_p=0.85, do_sample=True):
    # 将输入的prompt编码为tensor，并移动到指定设备上
    input_ids = torch.tensor(encode(prompt), dtype=torch.long)[None, :].to(device)
    # 克隆input_ids以避免修改原始输入
    idx = input_ids.clone()
    # 获取输入的长度
    input_len = input_ids.shape[1]

    # 循环生成新的token，直到达到最大长度或遇到结束符
    for _ in range(max_new_tokens):
        # 获取当前输入的最后block_size个token
        idx_cond = idx[:, -config.block_size:]
        # 使用模型生成logits
        logits = model(idx_cond)
        # 对logits进行缩放
        logits = logits[:, -1, :] / temperature
        # 计算softmax概率
        probs = F.softmax(logits, dim=-1)

        # 如果选择采样生成
        if do_sample:
            # 对概率进行降序排序
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            # 计算累积概率
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # 获取累积概率超过top_p的位置
            cutoff = cumulative_probs > top_p
            # 确保至少有一个token的概率不为0
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            # 将超过top_p的token概率设为0
            sorted_probs[cutoff] = 0
            # 归一化概率
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            # 使用多项式分布采样下一个token
            next_token = torch.multinomial(sorted_probs, num_samples=1)
            # 获取对应的token索引
            next_token = sorted_indices.gather(-1, next_token)
        else:
            # 如果不采样，选择概率最大的token
            next_token = torch.argmax(probs, dim=-1, keepdim=True)

        # 将生成的token添加到当前序列中
        idx = torch.cat((idx, next_token), dim=1)

        # 如果生成 <eos>，提前结束
        if next_token.item() == stoi.get("<eos>", -1):
            break

    # 只返回生成的部分（不包括 prompt）
    generated = idx[0, input_len:]
    return decode(generated.tolist())


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, config = load_model("transformer!/gpt/gpt_model.pt", device)

    while True:
        user_input = input("你：").strip()
        if user_input.lower() == "exit":
            break

        prompt = f"User: {user_input} Bot:"
        response = generate(model, config, prompt, device)
        print("🤖：" + response)
