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

elif run["GAN"]:
    pass

elif run["CGAN"]:
    pass

elif run["Diffusion"]:

    from models import diffusion_models

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