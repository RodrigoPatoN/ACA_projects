import torch
import os

run = {
    "VAE": True,
    "GAN": False,
    "CGAN": False,
    "Diffusion": False,
}

if run["VAE"]:

    from models import autoencoders 

    vae = autoencoders.VAE(color_channels=3, latent_dim=128)
    dae = autoencoders.DenoisingAutoencoder()

    vae.load_state_dict(torch.load("models/vae.pth"))
    dae.load_state_dict(torch.load("models/dae.pth"))

    # Generate images
    generated, refined = autoencoders.generate_images_vae_dae(vae, dae, num_images=100, latent_dim=128)

    # Save generated images
    for i in range(len(generated)):
        os.makedirs("generated_images", exist_ok=True)
        os.makedirs("generated_images/AE", exist_ok=True)
        os.makedirs("generated_images/AE/VAE", exist_ok=True)
        torch.save(generated[i], f"generated_images/AE/VAE/{i}.png")
        torch.save(refined[i], f"generated_images/AE/VAE_DAE/{i}.png")

    print("Generated images saved to 'generated_images/' directory.")