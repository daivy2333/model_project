import os
from PIL import Image
from torchvision import transforms

class ImageTextDataset:
    def __init__(self, image_folder, text_list, preprocess):

        # 初始化方法，接收三个参数：图片文件夹路径、文本列表和预处理函数
        self.image_folder= image_folder  # 存储图片文件夹路径
        # 获取图片文件夹中的所有图片路径，并存储在self.image_paths中
        self.image_paths= [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
        self.text_list= text_list     # 存储文本列表
        self.preprocess= preprocess  # 存储预处理函数


    def get_image_tensors(self):

        # 获取图片张量的方法
        images= []  # 初始化一个空列表，用于存储处理后的图片
        for path in self.image_paths:

            # 遍历所有图片路径
            image= Image.open(path).convert("RGB")  # 打开图片并转换为RGB格式
            image= self.preprocess(image)  # 对图片进行预处理
            images.append(image)  # 将处理后的图片添加到列表中

        return self.image_paths, images  # 返回图片路径列表和处理后的图片列表
    
    def get_texts(self):
        

        # 定义一个名为 get_texts 的方法，该方法属于某个类（由 self 参数表示）
        # 获取文本的方法
        return self.text_list  # 返回文本列表