import torch
import torch.utils.data.dataloader as torchdataloader
from torchvision import transforms, datasets, utils
import matplotlib as plt
import torch.optim as optim
from env_tf_2_3.pytorch_learn.alexnet.model import AlexNet
import os
import json
import time

MODEL_PATH = './model/net'


def dataprepare(batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_train = datasets.CIFAR10(root="./data/",
                                  transform=transform,
                                  train=True,
                                  download=True)
    data_test = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    train_loader = torchdataloader.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    test_loader = torchdataloader.DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def train(model: torch.nn.Module, criterion: torch.nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer,
          data_loader: torchdataloader.DataLoader, epochs: int = 10, log_per_time: int = 64):
    print("------train start----------")
    for epoch in range(epochs):
        running_loss = 0.0
        minAveLoss = 10000
        for step, (batch_x, batch_y) in enumerate(data_loader):
            # 计数增加
            step = step + 1

            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            output = model(batch_x)

            optimizer.zero_grad()
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % log_per_time == 0:
                aveloss = running_loss / log_per_time
                if aveloss < minAveLoss:
                    minAveLoss = aveloss
                    torch.save(model.state_dict(), MODEL_PATH)
                print("ave loss : %.4f" % aveloss)
                running_loss = 0

    print("Train Finished")


def trainning(batch_size: int = 64, learning_rate: float = 0.005):
    # 准备数据
    train_loader, test_loader = dataprepare(batch_size)
    # 类别标签
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # 创建网络
    alexnet = AlexNet()
    # print(alexnet)
    # 分配到GPU
    alexnet.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(alexnet.parameters(), lr=learning_rate)

    checkpoint = torch.load(MODEL_PATH)
    alexnet.load_state_dict(checkpoint)

    train(alexnet, criterion, optimizer, train_loader)


if __name__ == '__main__':
    trainning()
