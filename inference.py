import os
import torch
from library.models import AE, AutoEncoder, DeepAutoEncoder
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchsummary import summary    
from datetime import datetime
# My Modules
from library.utils import makedir


date = datetime.today().strftime('%Y%m%d_%H%M%S') 

PATH = './results/training/01_Examples/100_Epochs_linear_AE/model.pt'
dir_results = makedir('results')
dir_inference = makedir('inference', dir_results)
dir_run = makedir(f'run_{date}',dir_inference)

transform = ToTensor()
testset = MNIST(root='datasets/', train=False, transform=transform) 

input_size = testset[0][0].size()
input_size_flatten = input_size[1]*input_size[2]

model = torch.jit.load(PATH)
model.eval()
print(model)

fig, axs = plt.subplots(2, 10)

for i in range(10):
    data, _ = testset[i]
    data_flatten = torch.flatten(data,start_dim=1)
    recon = model(data_flatten)
    recon = recon.view(input_size[1], input_size[2])
    data = torch.squeeze(data)
    axs[0][i].imshow(data.detach().numpy())
    axs[0][i].axis('off')
    axs[1][i].imshow(recon.detach().numpy())
    axs[1][i].axis('off')
    plt.savefig(os.path.join(dir_run, 'results.svg'))