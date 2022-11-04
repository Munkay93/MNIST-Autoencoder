import torch
from library.models import AE, AutoEncoder, DeepAutoEncoder
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchsummary import summary    

PATH = './models/model_BCEWithLogitsLoss.pt'

transform = ToTensor()
testset = MNIST(root='./', train=False, transform=transform) 

input_size = testset[0][0].size()
input_size_flatten = input_size[1]*input_size[2]
embedding_size = 32
model = DeepAutoEncoder(input_dim=input_size_flatten, encoding_dim=embedding_size)
# model = AE(input_size=input_size_flatten, embedding_size=embedding_size)
model.load_state_dict(torch.load(PATH))

model.eval()
print(model)

for i in range(10):
    data, _ = testset[i]
    data_flatten = torch.flatten(data,start_dim=1)
    recon = model(data_flatten)
    recon = recon.view(input_size[1], input_size[2])
    data = torch.squeeze(data)
    fig, axs = plt.subplots(2,1)
    axs[0].imshow(data.detach().numpy())
    axs[1].imshow(recon.detach().numpy())
    plt.show()
