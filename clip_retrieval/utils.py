import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 计算相似度
def compute_similarity(image_features, text_features):
    return text_features @ image_features.T  # 返回 [1, N] 的矩阵


# 显示图片
def display_images(image_paths, indices, num_images):
    print(f"Image paths: {image_paths}")
    print(f"Indices: {indices}")

    for i in range(num_images):
        img_path = image_paths[indices[i].item()]
        img = Image.open(img_path)
        img.show()

