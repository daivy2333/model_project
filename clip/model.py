import clip
import torch

def load_clip_model():

    # 检查是否有可用的CUDA设备，如果有则使用CUDA，否则使用CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 使用clip库加载预训练的ViT-B/32模型，并将其加载到指定的设备上
    # 同时获取模型的预处理函数
    model, preprocess= clip.load("ViT-B/32", device=device)

    # 返回模型、预处理函数和设备信息
    return model, preprocess, device