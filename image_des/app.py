from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

# 加载BLIP模型和处理器
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    """
    根据输入的图像生成描述
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # 使用模型生成描述
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)

    return description

if __name__ == "__main__":
    image_path = input("请输入图片路径：")
    
    if os.path.exists(image_path):
        description = generate_caption(image_path)
        print(f"图像描述：{description}")
    else:
        print("图像文件不存在，请检查路径。")
