import torch
import torch.nn.functional as F
import torch.nn as nn

class Linear_AE(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super().__init__()
        self.encoder = Linear_Encoder(input_dim=input_dim, encoding_dim=encoding_dim)
        self.decoder = Linear_Decoder(output_dim=input_dim, encoding_dim=encoding_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Linear_Decoder(nn.Module):
    def __init__(self, encoding_dim, output_dim) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), 
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, encoded):
        decoded = self.decoder(encoded)
        return decoded
        
class Linear_Encoder(nn.Module):
    def __init__(self, input_dim, encoding_dim) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Linear(64, encoding_dim), 
            nn.ReLU(True),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

class CNN_AE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = CNN_Encoder()
        self.decoder = CNN_Decoder()
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class CNN_Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6*6*32, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded
    
class CNN_Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 6*6*32),
            nn.ReLU(),
            nn.Unflatten(1, (32, 6, 6)),
            nn.ConvTranspose2d(32, 16, 3, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2),
            nn.Sigmoid()
        )    
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded