from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import os

# 创建导出路径
os.makedirs("onnx_model", exist_ok=True)

# 加载模型和处理器
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

# 创建输入样本
from PIL import Image
image = Image.open("image_des_onnx\\assets\\美女的背影.png").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# 自定义封装模块：仅演示导出视觉编码部分
class VisionEncoder(torch.nn.Module):
    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values):
        return self.vision_model(pixel_values).last_hidden_state

vision_encoder = VisionEncoder(model.vision_model)

# 导出为 ONNX
torch.onnx.export(
    vision_encoder,
    inputs["pixel_values"],
    "onnx_model/model.onnx",
    input_names=["pixel_values"],
    output_names=["last_hidden_state"],
    dynamic_axes={"pixel_values": {0: "batch_size"}},
    opset_version=13,
    do_constant_folding=True
)
print("✅ 模型已成功导出为 onnx_model/model.onnx")