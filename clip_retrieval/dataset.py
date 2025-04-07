import os
from PIL import Image

class ImageTextDataset:
    def __init__(self, image_folder, preprocess):
        self.image_folder = image_folder
        self.preprocess = preprocess

    def get_image_tensors(self):
        # ✅ 每次都重新获取图像路径
        image_paths = [
            os.path.join(self.image_folder, f)
            for f in os.listdir(self.image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        images = []
        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")
                images.append(self.preprocess(image))
            except Exception as e:
                print(f"加载失败：{path}，错误：{e}")
        return image_paths, images
