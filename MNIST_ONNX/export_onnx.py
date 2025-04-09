# export_onnx.py
from fastai.vision.all import load_learner
import torch
import onnx

# 加载 fastai 模型
learn = load_learner('C:\\Users\\86187\\.fastai\\data\\mnist_png\\mnist_fastai.pkl')
model = learn.model.eval()

# 创建一个假的输入样本（示例大小和数据相同）
dummy_input = torch.randn(1, 3, 224, 224)

# 导出为 ONNX
torch.onnx.export(
    model,
    dummy_input,
    "mnist_model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=11
)
print("✅ ONNX 导出成功：mnist_model.onnx")
