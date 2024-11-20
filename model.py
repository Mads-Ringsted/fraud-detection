import torch.nn as nn

class AutoEncoder(nn.Module):
    
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28, 100),
            nn.ReLU(True),
            nn.Linear(100, 50),
            nn.ReLU(True),
            nn.Linear(50, 2)

        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(True),
            nn.Linear(50, 100),
            nn.ReLU(True),
            nn.Linear(100, 28),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

