import torch
from model import load_clip_model
from dataset import ImageTextDataset

import clip
import os

def main():

    # 加载CLIP模型及其预处理函数和设备（CPU或GPU）
    model, preprocess, device= load_clip_model()

    # 初始化文本列表，这里为空列表
    texts= ["一个熊猫", "一个只有背影的美女", "一张阿尔卑斯山的风景照"]

    # 创建图像文本数据集，传入图像文件夹路径、文本列表和预处理函数
    dataset= ImageTextDataset(image_folder="clip\images", text_list=texts, preprocess=preprocess)

    # 获取数据集中的图像路径和图像张量
    image_paths, image_tensors= dataset.get_image_tensors()

    # 将文本数据转换为CLIP模型可接受的令牌格式，并移动到指定设备
    text_tokens= clip.tokenize(dataset.get_texts()).to(device)

    # 将图像张量堆叠成一个批次，并移动到指定设备
    image_input= torch.stack(image_tensors).to(device)

    # 在不计算梯度的情况下执行以下操作
    with torch.no_grad():
        # 使用模型对图像进行编码，得到图像特征
        image_features= model.encode_image(image_input)
        # 使用模型对文本进行编码，得到文本特征
        text_features= model.encode_text(text_tokens)

    
    # 对图像特征和文本特征进行归一化处理
    image_features= image_features / image_features.norm(dim=-1, keepdim=True)
    text_features= text_features / text_features.norm(dim=-1, keepdim=True)

    # 计算图像特征和文本特征之间的相似度
    similarity= (100.0 * image_features @ text_features.T)

    # 获取相似度矩阵中每个图像与文本的最相似值及其索引
    values, indices= similarity.topk(1, dim=1)

    # 打印结果标题
    print("结果")

    # 遍历图像路径，打印每个图像的匹配文本及其相似度
    for i, path in enumerate(image_paths):
        print(f"图片 {path} => 匹配文本：{texts[indices[i].item()]}，相似度：{values[i].item():.2f}")
if __name__ == "__main__":
    main()