import os
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as F

import medmnist
from medmnist import INFO, Evaluator

from .autoencoders import Autoencoder_Linear


# -------------------------------- Load Dataset --------------------------------

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

data_flag = 'bloodmnist'
download = True
BATCH_SIZE = 128

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])


# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load all splits
train = DataClass(split='train', transform=data_transform, download=True)
val = DataClass(split='val', transform=data_transform, download=True)
test = DataClass(split='test', transform=data_transform, download=True)

# Combine them into a single dataset
full_dataset = ConcatDataset([train, val, test])
dataloader = DataLoader(full_dataset, batch_size=128, shuffle=True)


# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load all splits
train = DataClass(split='train', transform=data_transform, download=True)
val = DataClass(split='val', transform=data_transform, download=True)
test = DataClass(split='test', transform=data_transform, download=True)

# Combine them into a single dataset
full_dataset = ConcatDataset([train, val, test])
dataloader = DataLoader(full_dataset, batch_size=128, shuffle=True)

# -------------------------------- Setup --------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

run = {
    "Autoencoder": False,
    "GAN": True,
    "Diffusion": False,
}

# -------------------------------- Autoencoder --------------------------------

# Code for autoencoder will go here

if run["Autoencoder"]:
    pass

# -------------------------------- GAN --------------------------------

if run["GAN"]:

    from .models.gans import Generator, Discriminator
    from .models.gans import generate_images
    from .models.gans import save_image

    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # still need to define whether i will touch the hyperparameters or not
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    latent_dim = 100
    batch_size = 128
    num_epochs = 20
    k = 1

    real_label = 1
    fake_label = 0

    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        for i, (data, _) in enumerate(dataloader):

            for _ in range(k):
                
                # Step 1: Train on real images
                netD.zero_grad()
                
                real_images = data.to(device)
                b_size = real_images.size(0)
                real_labels = torch.full((b_size,), real_label, dtype=torch.float, device=device)

                output_real = netD(real_images)
                loss_real = criterion(output_real, real_labels)
                loss_real.backward()

                # Step 2: Train on fake images
                noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
                fake_images = netG(noise).detach()
                fake_labels = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

                output_fake = netD(fake_images)
                loss_fake = criterion(output_fake, fake_labels)
                loss_fake.backward()

                # Step 3: Update Discriminator
                lossD = loss_real + loss_fake
                optimizerD.step()

            ########################
            # 2. Train Generator
            ########################
            netG.zero_grad()

            # (Step 5) Sample new noise
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)

            # (Step 6) Generate images and train G to fool D
            fake_images = netG(noise)
            labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)  # trick D

            output = netD(fake_images)
            lossG = criterion(output, labels)
            lossG.backward()
            optimizerG.step()

            # Optional: print log
            if i % 40 == 0:
                print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {lossD.item():.4f} Loss_G: {lossG.item():.4f}")

        # Save generated samples at the end of each epoch
        with torch.no_grad():
            fixed_noise = torch.randn(5, latent_dim, 1, 1, device=device)
            fake = netG(fixed_noise).detach().cpu()
            os.makedirs('output', exist_ok=True)
            save_image(fake, f'output/fake_samples_epoch_{epoch:03d}.png', normalize=True)

    print("Training complete.")