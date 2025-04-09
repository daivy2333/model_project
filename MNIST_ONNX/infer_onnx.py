# infer_onnx.py
import onnxruntime as ort
import torch
from torchvision import transforms
from PIL import Image

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 加载图像
img = Image.open("MNIST_ONNX\\some_digit.png").convert("RGB")
img_tensor = transform(img).unsqueeze(0).numpy()

# 推理
session = ort.InferenceSession("D:\\data_store\\onnx_model\\mnist_model.onnx")
outputs = session.run(None, {'input': img_tensor})
pred = torch.tensor(outputs[0]).argmax(dim=1).item()

print(f"🔢 预测结果：{pred}")
