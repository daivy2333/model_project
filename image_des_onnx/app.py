import onnxruntime as ort
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import numpy as np

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 ONNX 模型
ort_session = ort.InferenceSession("D:\\data_store\\onnx_model\\model.onnx")

# 加载原始模型和处理器
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

# 加载图像
image = Image.open("image_des_onnx\\assets\\美女的背影.png").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"].cpu().numpy()

# 使用 ONNX 提取视觉特征
onnx_outputs = ort_session.run(None, {"pixel_values": pixel_values})
encoder_hidden_states = torch.tensor(onnx_outputs[0]).to(device)

# 使用 text_decoder 生成描述
generated_ids = model.text_decoder.generate(
    encoder_hidden_states=encoder_hidden_states,
    attention_mask=torch.ones(encoder_hidden_states.shape[:2], dtype=torch.long).to(device),
    max_length=50
)

# 解码
caption = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"🖼️ 图像描述：{caption}")
