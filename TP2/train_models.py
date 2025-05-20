run = {
    "VAE": True,
    "DAE": True,
    "GAN": False,
    "CGAN": False,
    "Diffusion": False,
}

# -------------------------------- Autoencoder --------------------------------

chosen = False


while not chosen:

    print("SELECT MODEL TO TRAIN:\n\n1.VAE\n2.DAE\n3.GAN\n4.CGAN\n5.Diffusion\n6.ALL\n")
    model_choice = input("Enter your choice (1/2/3/4/5): ")
    chosen = True

    if model_choice == "1":

        run["VAE"] = True
        run["DAE"] = False
        run["GAN"] = False
        run["CGAN"] = False
        run["Diffusion"] = False

    elif model_choice == "2":

        run["VAE"] = False
        run["DAE"] = True
        run["GAN"] = False
        run["CGAN"] = False
        run["Diffusion"] = False

    elif model_choice == "3":

        run["VAE"] = False
        run["DAE"] = False
        run["GAN"] = True
        run["CGAN"] = False
        run["Diffusion"] = False

    elif model_choice == "4":

        run["VAE"] = False
        run["DAE"] = False
        run["GAN"] = False
        run["CGAN"] = True
        run["Diffusion"] = False

    elif model_choice == "5":

        run["VAE"] = False
        run["DAE"] = False
        run["GAN"] = False
        run["CGAN"] = False
        run["Diffusion"] = True

    elif model_choice == "6":

        run["VAE"] = True
        run["DAE"] = True
        run["GAN"] = True
        run["CGAN"] = True
        run["Diffusion"] = True

    else:
        print("Invalid choice.")
        chosen = False

# -------------------------------- Load Dataset --------------------------------

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

import medmnist
from medmnist import INFO

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
#data_transform = transforms.Compose([
#    transforms.ToTensor(),
#])

data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
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
SEEDS = [42, 123, 2024, 7, 888]

for seed_num, seed in enumerate(SEEDS):

    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(f'trained_models/', exist_ok=True)
    os.makedirs(f'trained_models/{seed}/', exist_ok=True)
    os.makedirs(f'output_training/', exist_ok=True)
    os.makedirs(f'output_training/{seed}/', exist_ok=True)
    os.makedirs(f"losses/", exist_ok=True)
    os.makedirs(f"losses/{seed}/", exist_ok=True)

    print(f"Using seed {seed} for random sampling.")

    if run["VAE"]:
        
        from models import autoencoders

        learning_rate = 0.001

        vae = autoencoders.VAE(color_channels=3, latent_dim=128)
        autoencoders.train_vae(vae, dataloader, seed, device=device, epochs=300, learning_rate=learning_rate)     

        torch.save(vae.state_dict(), f'trained_models/{seed}/vae_model_{learning_rate}.pth')
        print("VAE model saved.")

    if run["DAE"]:
        
        from models import autoencoders

        learning_rate = 0.001

        dae = autoencoders.DenoisingAutoencoder()
        dae = autoencoders.train_dae(dae, dataloader, seed, device=device, epochs=300, learning_rate=learning_rate)

        torch.save(dae.state_dict(), f'trained_models/{seed}/dae_model_{learning_rate}.pth')
        print("DAE model saved.")

    # -------------------------------- GAN --------------------------------

    if run["GAN"]:

        from models import gans
        from torchvision.utils import save_image

        netG = gans.Generator().to(device)
        netD = gans.Discriminator().to(device)

        learning_rate = 0.001

        # still need to define whether i will touch the hyperparameters or not
        optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        criterion = nn.BCELoss()

        latent_dim = 100
        batch_size = 128
        num_epochs = 300
        k = 1

        real_label = 1
        fake_label = 0

        criterion = nn.BCELoss()

        losses = []

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

            print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {lossD.item():.4f} Loss_G: {lossG.item():.4f}")
            losses.append(lossG.item())

            # Save generated samples at the end of each epoch
            with torch.no_grad():
                fixed_noise = torch.randn(5, latent_dim, 1, 1, device=device)
                fake = netG(fixed_noise).detach().cpu()

                os.makedirs(f'output_training/{seed}/DCGAN/', exist_ok=True)
                os.makedirs(f'output_training/{seed}/DCGAN/{learning_rate}/', exist_ok=True)

                save_image(fake, f'output_training/{seed}/DCGAN/{learning_rate}/fake_samples_epoch_{epoch:03d}.png', normalize=True)

        # save losses
        with open(f'losses_GAN_{learning_rate}.txt', 'w') as f:
            for loss in losses:
                f.write(f"{loss}\n")

        print("Training complete.")

        torch.save(netG.state_dict(), f'./trained_models/{seed}/GAN_netG_{learning_rate}.pth')
        torch.save(netD.state_dict(), f'./trained_models/{seed}/GAN_netD_{learning_rate}.pth')

    if run["CGAN"]:

        from models import cgans
        from torchvision.utils import save_image

        netG = cgans.Generator(num_classes=8).to(device)
        netD = cgans.Discriminator(num_classes=8).to(device)

        learning_rate = 0.001

        optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        criterion = nn.BCELoss()

        latent_dim = 100
        batch_size = 128
        num_epochs = 300
        k = 1

        real_label = 1
        fake_label = 0

        criterion = nn.BCELoss()

        losses = []

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


            print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {lossD.item():.4f} Loss_G: {lossG.item():.4f}")
            losses.append(lossG.item())
            # Save generated samples at the end of each epoch
            with torch.no_grad():
                fixed_noise = torch.randn(5, latent_dim, 1, 1, device=device)
                fixed_labels = torch.randint(3, 5, (5,), device=device)
                fake = netG(fixed_noise, fixed_labels).detach().cpu()

                os.makedirs(f'output_training/{seed}/CGAN/', exist_ok=True)
                os.makedirs(f'output_training/{seed}/CGAN/{learning_rate}/', exist_ok=True)

                save_image(fake, f'output_training/{seed}/CGAN/{learning_rate}/fake_samples_epoch_{epoch:03d}.png', normalize=True)

        print("Training complete.")

            # save losses
        with open(f'losses_GAN_{learning_rate}.txt', 'w') as f:
            for loss in losses:
                f.write(f"{loss}\n")

        torch.save(netG.state_dict(), f'./trained_models/{seed}/CGAN_netG_{learning_rate}.pth')
        torch.save(netD.state_dict(), f'./trained_models/{seed}/CGAN_netD_{learning_rate}.pth')

    # -------------------------------- Diffusion --------------------------------

    if run["Diffusion"]:

        from models import diffusion_models 
        from torch.optim import Adam

        no_train = False
        batch_size = 128
        n_epochs = 300
        learning_rate = 0.001

        # Defining model
        n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors of the paper
        ddpm = diffusion_models.MyDDPM(diffusion_models.MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)

        print(sum([p.numel() for p in ddpm.parameters()]))

        # Training
        store_path = f"./trained_models/{seed}/diff_model_{learning_rate}.pt"
        if not no_train:
            diffusion_models.training_loop(ddpm,
                                        dataloader, 
                                        n_epochs, 
                                        optim=Adam(ddpm.parameters(), learning_rate), 
                                        device=device, 
                                        store_path=store_path, 
                                        learning_rate=learning_rate)