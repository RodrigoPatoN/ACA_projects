import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(Generator, self).__init__()
        n_dim_class = 100
        self.label_emb = nn.Embedding(num_classes, n_dim_class)
        #self.label_emb = nn.Embedding(num_classes, latent_dim)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + n_dim_class, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, noise, labels):
        labels = labels.long()
        label_embedding = self.label_emb(labels).unsqueeze(2).unsqueeze(3)  # shape: [B, C, 1, 1]
        input = torch.cat((noise, label_embedding), dim=1)
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, num_classes=8):
        super(Discriminator, self).__init__()
        n_dim_class = 100
        self.label_emb = nn.Embedding(num_classes, n_dim_class)

        self.model = nn.Sequential(
            nn.Conv2d(3 + n_dim_class, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(0.3),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
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
    with torch.no_grad():

        noise = torch.randn(num_images, 100, 1, 1, device=device)

        class_probs = np.array([1218, 3117, 1551, 2895, 1214, 1420, 3329, 2348], dtype=np.float32)
        class_probs /= class_probs.sum()

        labels = torch.tensor(np.random.choice(8, size=num_images, p=class_probs), device=device)

        generated_images = generator(noise, labels)
        #generated_images = (generated_images + 1) / 2  # Rescale images from [-1, 1] to [0, 1]
        return generated_images
