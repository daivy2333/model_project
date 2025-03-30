from unet_model import UNet
from dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch

def train_net(net, device, data_path, epochs= 40, batch_size= 1, lr= 0.00001):

    isbi_dataset= ISBI_Loader(data_path)
    train_loader= torch.utils.data.DataLoader(dataset= isbi_dataset, batch_size= batch_size, shuffle= True)
    optimizer= optim.RMSprop(net.parameters(), lr= lr, weight_decay= 1e-8, momentum= 0.9)

    criterion= nn.BCEWithLogitsLoss()

    best_loss= float('inf')

    for epoch in range(epochs):
        net.train()

        for image, label in train_loader:
            image, label= image.to(device), label.to(device)
            image = image.float()  # 转换图像数据为浮点数类型
            label = label.float()
            optimizer.zero_grad()

            pred= net(image)
            loss= criterion(pred, label)
            print(loss.item())

            if loss < best_loss:
                best_loss= loss
                torch.save(net.state_dict(), 'best_model.pth')

            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net= UNet(n_channels= 1, n_classes= 1).to(device)
    train_net(net, device, data_path= 'u-net\data\\train')
