from statistics import mode
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.nn as nn
from models import AE
from utils import train, validation

transform = ToTensor()

trainset = MNIST(root='./', train=True, download=True, transform=transform)
testset = MNIST(root='./', train=False, transform=transform)

trainloader = DataLoader(dataset=trainset, batch_size=64)
testloader = DataLoader(dataset=testset, batch_size=64)

input_size = trainloader.dataset[0][0].size()
input_size = input_size[1]*input_size[2]

model = AE(input_size=input_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters() ,lr=1e-3)

num_epochs = 10

for epoch in range(num_epochs):
    train(model=model, trainloader=trainloader, criterion=criterion, optimizer=optimizer)

    model.eval()
    for batch_input in testloader:
        imgs, _ = batch_input
        imgs = torch.flatten(imgs, start_dim=1)
        recons = model(imgs)
        loss = criterion(imgs, recons)

PATH = './models/model.pt'
torch.save(model.state_dict(), PATH)


