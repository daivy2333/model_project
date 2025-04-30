import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
from utils import build_vocab, encode, decode
from pathlib import Path

def load_model(model_path, device, block_size=128):
    text = Path("transformer!\gpt\input.txt").read_text(encoding='utf-8')
    vocab_size = build_vocab(text)
    config = GPTConfig(vocab_size, block_size=block_size)
    model = GPT(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, config

@torch.no_grad()
def generate(model, config, prompt, device, max_new_tokens=20):
    idx = torch.tensor(encode(prompt), dtype=torch.long)[None, :].to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -config.block_size:]
        logits = model(idx_cond)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

        # 解码目前为止的全部文本
        text = decode(idx[0].tolist())
        new_text = text[len(prompt):]

        # 如果检测到 'Q:'，就截断，不保留 'Q:' 本身
        if "Q:" in new_text:
            cutoff = new_text.index("Q:")
            new_text = new_text[:cutoff]
            return prompt + new_text.strip()

    return decode(idx[0].tolist())


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, config = load_model("gpt_model.pt", device)
    while(1):
        prompt = "Q: " + input("请输入问题(exit退出)：") + "\nA:"
        if prompt == "Q: exit\nA:":
            break
        response = generate(model, config, prompt, device, max_new_tokens=20)
        print(response)
    
