from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch

# 自动下载模型
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32
)

# 替换 scheduler 以设置步数
new_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
new_scheduler.config.num_inference_steps = 25
pipe.scheduler = new_scheduler

pipe.set_progress_bar_config(disable=True)

# 生成图像
prompt = "A cat sitting on a spaceship, digital art"
image = pipe(prompt=prompt, height=256, width=256).images[0]
image.save("generated_image.png")
print("✅ 图像已生成并保存为 generated_image.png")

# 导出 UNet 为 ONNX（推理部分）
unet = pipe.unet.eval()
dummy_input = {
    "sample": torch.randn(1, 4, 64, 64),
    "timestep": torch.tensor([1]),
    "encoder_hidden_states": torch.randn(1, 77, 768)
}
torch.onnx.export(
    unet,
    (dummy_input["sample"], dummy_input["timestep"], dummy_input["encoder_hidden_states"]),
    "sd_unet.onnx",
    input_names=["sample", "timestep", "encoder_hidden_states"],
    output_names=["out_sample"],
    opset_version=15,
    dynamic_axes={"sample": {0: "batch"}, "encoder_hidden_states": {0: "batch"}}
)
print("✅ UNet 模块已导出为 ONNX 格式：sd_unet.onnx")
