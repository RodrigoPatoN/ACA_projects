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

# -------------------------------- Setup --------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

run = {
    "VAE": True,
    "GAN": False,
    "CGAN": False,
    "Diffusion": False,
}

# -------------------------------- Autoencoder --------------------------------

# Code for autoencoder will go here

if run["VAE"]:
    
    from .models import autoencoders

    vae = autoencoders.VAE(color_channels=3, latent_dim=128)
    autoencoders.train_vae(vae, dataloader, device=device, epochs=30)     

    dae = autoencoders.DenoisingAutoencoder()
    dae = autoencoders.train_dae(dae, dataloader, device=device, epochs=20)

    torch.save(vae.state_dict(), 'vae_model.pth')
    torch.save(dae.state_dict(), 'dae_model.pth')
    print("VAE and DAE models saved.")

    #vae_out, dae_out = autoencoders.generate_images_vae_dae(vae, dae, num_images=16, latent_dim=128, device=device)
    #plot_images(vae_out, dae_out)

# -------------------------------- GAN --------------------------------

if run["GAN"]:

    from .models import gans
    from torchvision.utils import save_image

    netG = gans.Generator().to(device)
    netD = gans.Discriminator().to(device)

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

    torch.save(netG.state_dict(), './netG.pth')
    torch.save(netD.state_dict(), './netD.pth')

if run["CGAN"]:

    from .models import cgans

    netG = cgans.Generator().to(device)
    netD = cgans.Discriminator().to(device)

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

    latent_dim = 100
    batch_size = 128
    num_epochs = 100
    k = 1

    real_label = 1
    fake_label = 0

    for epoch in range(num_epochs):
        for i, (data, label) in enumerate(dataloader):

            for _ in range(k):
                
                # Step 1: Train on real images
                netD.zero_grad()
                
                real_images = data.to(device)
                real_class_labels = label.to(device).long()  # For embedding
                b_size = real_images.size(0)
                real_labels = torch.full((b_size,), real_label, dtype=torch.float, device=device)  # For BCELoss

                output_real = netD(real_images, real_class_labels)
                loss_real = criterion(output_real, real_labels)
                loss_real.backward()

                # Step 2: Train on fake images
                noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
                fake_class_labels = torch.randint(3, 5, (b_size,), device=device)
                fake_images = netG(noise, fake_class_labels)
                output_fake = netD(fake_images.detach(), fake_class_labels)
                fake_targets = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

                loss_fake = criterion(output_fake, fake_targets)
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
            gen_labels = torch.randint(3, 5, (batch_size,), device=device)
            fake_images = netG(noise, gen_labels)
            output = netD(fake_images, gen_labels)
            valid = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            lossG = criterion(output, valid)
            lossG.backward()
            optimizerG.step()

            # Optional: print log
            if i % 40 == 0:
                print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {lossD.item():.4f} Loss_G: {lossG.item():.4f}")

        # Save generated samples at the end of each epoch
        with torch.no_grad():
            fixed_noise = torch.randn(5, latent_dim, 1, 1, device=device)
            fixed_labels = torch.randint(3, 5, (5,), device=device)
            fake = netG(fixed_noise, fixed_labels).detach().cpu()
            os.makedirs('output', exist_ok=True)
            save_image(fake, f'output/fake_samples_epoch_{epoch:03d}.png', normalize=True)

    print("Training complete.")

    torch.save(netG.state_dict(), './netG_CGAN.pth')
    torch.save(netD.state_dict(), './netD_CGAN.pth')

# -------------------------------- Diffusion --------------------------------

if run["Diffusion"]:

    from .models import diffusion_models 
    from torch.optim import Adam

    no_train = False
    batch_size = 128
    n_epochs = 10
    lr = 0.001

    # Defining model
    n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors of the paper
    ddpm = diffusion_models.MyDDPM(diffusion_models.MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)

    print(sum([p.numel() for p in ddpm.parameters()]))

    # Training
    store_path = "diff_model.pt"
    if not no_train:
        diffusion_models.training_loop(ddpm, dataloader, n_epochs, optim=Adam(ddpm.parameters(), lr), device=device, store_path=store_path)

    best_model = diffusion_models.MyDDPM(diffusion_models.MyUNet(), n_steps=n_steps, device=device)
    best_model.load_state_dict(torch.load(store_path, map_location=device))
    best_model.eval()
    print("Model loaded")

    print("Generating new images")
    generated = diffusion_models.generate_new_images(
            best_model,
            option = 1,
            n_samples=16,# change the number of samples as needed
            device=device,
            gif_name="test.gif"
        )
    #diffusion_models.show_images(generated, "Final Option 1 result")


    print("Generating new images")
    generated = diffusion_models.generate_new_images(
            best_model,
            option = 1,
            n_samples=16,# change the number of samples as needed
            device=device,
            gif_name="test.gif"
        )
    
    #diffusion_models.show_images(generated, "Final Option 1 result")