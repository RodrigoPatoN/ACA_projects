import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np



class Autoencoder_Linear(nn.Module):
    def __init__(self, color_channels=1, width=28, height=28):
        super().__init__()        
        self.target_width = width
        self.target_height = height
        self.target_color_channels = color_channels
        self.encoder = nn.Sequential(
            nn.Linear(self.target_color_channels * self.target_height * self.target_width, 128), # (BatchSize, Width*Height) -> (BatchSize, 128)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )
        

        self.decoder = nn.Sequential(
            #TODO: Implement the decoder part of the autoencoder
            #Remember that the activation function of the last layer 
            #depends on the range of the input values
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    


def train(model, data_loader, num_epochs=50, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=1e-3, 
                                weight_decay=1e-5)


    model = model.to(device)
    
    outputs = []
    for epoch in range(num_epochs):
        for (img, _) in data_loader:
            #TODO: Implement the training loop
            #Remember to move the data to the device and that
            #the training loop should be similar to the one of
            #a regular neural network
            pass
            
            
        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
        outputs.append((epoch, img.to('cpu'), recon.to('cpu')))
        
    
    return model.to('cpu'), outputs

