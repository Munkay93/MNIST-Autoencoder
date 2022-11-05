import torch

def train(model, trainloader, criterion, optimizer):    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    for batch_input in trainloader:
        imgs, _ = batch_input
        imgs = imgs.to(device)
        recons = model(imgs)
        loss = criterion(imgs, recons)
        loss = criterion(recons, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validation(model, loader, criterion):
    model.eval()
    loss_batch = 0.0
    for batch_input in loader:
        imgs, _ = batch_input
        imgs = imgs.to(device='cuda')
        recons = model(imgs)
        loss = criterion(imgs, recons)
        loss_batch += loss.item()

    final_loss = loss_batch/len(loader)
    return final_loss    