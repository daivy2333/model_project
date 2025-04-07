import os
import torch
import clip
from dataset import ImageTextDataset
from utils import compute_similarity, display_images

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def main():
    model, preprocess, device = load_clip_model()
    dataset = ImageTextDataset("clip_retrieval\images", preprocess)

    while True:
        user_input = input("请输入文本提示（输入 q 退出）：")
        if user_input.lower() == "q":
            break

        # 🔁 每次都重新获取图像（确保变化！）
        image_paths, image_tensors = dataset.get_image_tensors()
        print("🖼 当前图像路径：", image_paths)
        print("🎯 图像张量均值：", torch.stack(image_tensors).mean())

        image_input = torch.stack(image_tensors).to(device)

        with torch.no_grad():
            # 编码图像
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # 编码文本
            text_tokens = clip.tokenize([user_input]).to(device)
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        # 输出前5相似
        # 输出前k相似
            similarity = compute_similarity(image_features, text_features)
            similarity = similarity.squeeze(0)  # 压缩多余的维度
            k = min(5, len(image_paths))  # 确保 k 不大于图像数量

            values, indices = similarity.topk(k, dim=0)
            # 确保 indices 只包含一维索引
            # 如果 indices 是二维的，则执行 squeeze(1)
            if indices.dim() > 1:
                indices = indices.squeeze(1)

            print(f"top {k} indices: {indices}")
            print("similarity shape:", similarity.shape)  # 检查相似度矩阵
            print("similarity values:", similarity)       # 输出相似度的值



        print(f"\n📌 最相关的 {k} 张图片如下：")
        display_images(image_paths, indices[0], num_images=k)

        # Debug 查看变化
        # 找出最相似图的 index
        best_match_idx = indices[0][0].item()

        # 输出匹配到的那张图像的特征，而不是第0张图像
        print("text_features:", text_features[0][:5])
        print("image_features:", image_features[best_match_idx][:5])


if __name__ == "__main__":
    main()
