import torch.nn as nn

class AE(nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.input_size = input_size

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, int(input_size/2)),
            nn.ReLU(),
            nn.Linear(int(input_size/2), int(input_size/4)),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(int(input_size/4), int(input_size/2)),
            nn.ReLU(),
            nn.Linear(int(input_size/2), int(input_size)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded