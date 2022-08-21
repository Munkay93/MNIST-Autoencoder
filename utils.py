import torch

def train(model, trainloader, criterion, optimizer):
    
    model.train()
    for batch_input in trainloader:
        imgs, _ = batch_input
        imgs = torch.flatten(imgs, start_dim=1)
        recons = model(imgs)
        loss = criterion(imgs, recons)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validation():
    pass