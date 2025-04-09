# mnist_train.py
from fastai.vision.all import *

# 加载 MNIST 数据集
path = untar_data(URLs.MNIST)
dls = ImageDataLoaders.from_folder(path, train='training', valid='testing', bs=64, num_workers=0)

# 定义模型（使用 resnet18）
learn = vision_learner(dls, resnet18, metrics=accuracy)

# 训练模型
learn.fine_tune(3)

# 保存模型
learn.export('mnist_fastai.pkl')
