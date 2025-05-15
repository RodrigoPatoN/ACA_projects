import torch
import os
from torchvision.utils import save_image

run = {
    "VAE": True,
    "GAN": False,
    "CGAN": False,
    "Diffusion": False,
}

num_images = 10000

if run["VAE"]:

    from models import autoencoders 

    vae = autoencoders.VAE(color_channels=3, latent_dim=128)
    dae = autoencoders.DenoisingAutoencoder()

    vae.load_state_dict(torch.load("models/vae.pth"))
    dae.load_state_dict(torch.load("models/dae.pth"))

    # Generate images
    generated, refined = autoencoders.generate_images_vae_dae(vae, dae, num_images=num_images, latent_dim=128)

    # Save generated images
    for i in range(len(generated)):
        os.makedirs("generated_images", exist_ok=True)
        os.makedirs("generated_images/AE", exist_ok=True)
        os.makedirs("generated_images/AE/VAE", exist_ok=True)
        os.makedirs("generated_images/AE/VAE_DAE", exist_ok=True)
        save_image(generated[i], f"generated_images/AE/VAE/{i}.png")
        save_image(refined[i], f"generated_images/AE/VAE_DAE/{i}.png")

    print("Generated images saved to 'generated_images/' directory.")