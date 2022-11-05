import torch

def train(model, trainloader, criterion, optimizer, flatten):    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    for batch_input in trainloader:
        imgs, _ = batch_input
        imgs = imgs.to(device)
        if flatten:
            imgs = torch.flatten(imgs, start_dim=1)
        recons = model(imgs)
        loss = criterion(imgs, recons)
        loss = criterion(recons, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validation(model, loader, criterion, flatten):
    model.eval()
    loss_batch = 0.0
    for batch_input in loader:
        imgs, _ = batch_input
        imgs = imgs.to(device='cuda')
        if flatten:
            imgs = torch.flatten(imgs, start_dim=1)
        recons = model(imgs)
        loss = criterion(imgs, recons)
        loss_batch += loss.item()

    final_loss = loss_batch/len(loader)
    return final_loss    