import os
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchsummary import summary    
from datetime import datetime
# My Modules
from library.utils import makedir


date = datetime.today().strftime('%Y%m%d_%H%M%S') 

PATH = './results/training/01_Examples/CNN_AE_example/model.pt'
dir_results = makedir('results')
dir_inference = makedir('inference', dir_results)

transform = ToTensor()
testset = MNIST(root='datasets/', train=False, transform=transform) 

input_size = testset[0][0].size()
input_size_flatten = input_size[1]*input_size[2]

model = torch.jit.load(PATH)
model.eval()
print(model)

dir_run = makedir(f'{model.original_name}_{date}',dir_inference)

fig = plt.figure()
subfigs = fig.subfigures(2)
subfigs[0].suptitle('Ground Truth')
subfigs[1].suptitle('Reconstruction')
axs_row_1 = subfigs[0].subplots(1, 10)
axs_row_2 = subfigs[1].subplots(1, 10)

for i in range(10):
    data, _ = testset[i]
    data = data[None, :,:,:]
    recon = model(data)
    recon = recon.view(input_size[1], input_size[2])
    data = torch.squeeze(data)
    axs_row_1[i].imshow(data.detach().numpy())
    axs_row_1[i].axis('off')
    axs_row_2[i].imshow(recon.detach().numpy())
    axs_row_2[i].axis('off')
    plt.savefig(os.path.join(dir_run, 'results.svg'))