import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + num_classes, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        labels = labels.long()
        label_embedding = self.label_emb(labels).unsqueeze(2).unsqueeze(3)  # shape: [B, C, 1, 1]
        input = torch.cat((noise, label_embedding), dim=1)
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Conv2d(3 + num_classes, 64, 4, 2, 1, bias=False),  # Input now has extra channels
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(512, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        
        if labels.dim() == 2 and labels.size(1) == 1:
            labels = labels.squeeze(1)

        label_embedding = self.label_emb(labels)
        label_embedding = label_embedding.unsqueeze(2).unsqueeze(3)
        label_embedding = label_embedding.expand(-1, -1, img.size(2), img.size(3))
        input = torch.cat((img, label_embedding), dim=1)
        
        return self.model(input).view(-1, 1).squeeze(1)

def generate_images(generator, num_images, device):
    with torch.no_grad():  # Temporarily set all the requires_grad flag to false
        noise = torch.randn(num_images, 100, 1, 1, device=device)  # 100 is the size of the noise vector
        generated_images = generator(noise)
        generated_images = (generated_images + 1) / 2  # Rescale images from [-1, 1] to [0, 1]
        return generated_images