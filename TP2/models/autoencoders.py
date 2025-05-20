import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class VAE(nn.Module):
    def __init__(self, color_channels=1, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(color_channels, 16, 3, stride=2, padding=1),  # (B,16,16,16)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # (B,32,8,8)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # (B,64,4,4)
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(64 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 64 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # (B,32,8,8)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # (B,16,16,16)
            nn.ReLU(),
            nn.ConvTranspose2d(16, color_channels, 4, stride=2, padding=1),  # (B,3,32,32)
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 64, 4, 4)
        return self.decoder(x)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    

def vae_loss_function(recon_x, x, mu, log_var):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss


def train_vae(model, dataloader, learning_rate=0.001, device='cpu', epochs=300):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_kl_loss = 0
        total_recon_loss = 0
        for x, _ in dataloader:
            x = x.to(device)
            recon, mu, logvar = model(x)

            # Loss: MSE + KL
            recon_loss = F.mse_loss(recon, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            beta = min(1.0, epoch / 100)
            loss = recon_loss + beta * kl_loss

            #loss = recon_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_kl_loss += kl_loss.item()
            total_recon_loss += recon_loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] VAE Loss: {total_loss/len(dataloader.dataset):.4f} - Reconstruction Loss: {total_recon_loss/len(dataloader.dataset):.4f} - KL Loss: {total_kl_loss/len(dataloader.dataset):.4f}")

        losses.append(total_loss / len(dataloader.dataset))

    # save losses
    with open(f"losses/{torch.seed}/vae_losses_{learning_rate}.txt", "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")

    return model.cpu()


class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


def train_dae(model, dataloader, device='cpu', learning_rate=0.01, epochs=300):

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(epochs):
        print(epoch)
        model.train()
        total_loss = 0
        for x, _ in dataloader:
            x = x.to(device)
            noise = torch.randn_like(x)
            #noise = torch.clamp(noise, 0., 1.)
            out = model(noise)
            loss = criterion(out, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] DAE Loss: {total_loss/len(dataloader):.4f}")
        losses.append(total_loss / len(dataloader))
    
    with open(f"./losses/{torch.seed}/dae_losses_{learning_rate}.txt", "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")

    return model.cpu()


def generate_images_vae(vae, num_images=16, latent_dim=128, random_seed=8, device='cpu'):

    vae.eval()
    vae.to(device)
    
    with torch.no_grad():
        g = torch.Generator(device=device).manual_seed(random_seed)
        z = torch.randn((num_images, latent_dim), generator=g, device=device)
        z = torch.randn(num_images, latent_dim, random_seed=random_seed).to(device)
        generated = vae.decode(z)
    return generated.cpu()