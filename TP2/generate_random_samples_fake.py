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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chosen = False


while not chosen:

    print("SELECT MODEL TO TRAIN:\n\n1. VAE\n2.GAN\n3.CGAN\n4.Diffusion\n5.ALL\n")
    model_choice = input("Enter your choice (1/2/3/4/5): ")
    chosen = True

    if model_choice == "1":

        run["VAE"] = True
        run["GAN"] = False
        run["CGAN"] = False
        run["Diffusion"] = False

    elif model_choice == "2":

        run["VAE"] = False
        run["GAN"] = True
        run["CGAN"] = False
        run["Diffusion"] = False

    elif model_choice == "3":

        run["VAE"] = False
        run["GAN"] = False
        run["CGAN"] = True
        run["Diffusion"] = False

    elif model_choice == "4":

        run["VAE"] = False
        run["GAN"] = False
        run["CGAN"] = False
        run["Diffusion"] = True

    elif model_choice == "5":

        run["VAE"] = True
        run["GAN"] = True
        run["CGAN"] = True
        run["Diffusion"] = True

    else:
        print("Invalid choice.")
        chosen = False
        

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

    print("Generated AE images saved to 'generated_images/' directory.")

elif run["GAN"]:

    from models import gans

    generator = gans.Generator()
    generator.load_state_dict(torch.load("models/GAN_netG.pth", map_location=device))

    generator.to(device)

    images = gans.generate_images(generator, num_images=num_images, device=device)

    # Save generated images
    for i in range(len(images)):
        os.makedirs("generated_images", exist_ok=True)
        os.makedirs("generated_images/GAN", exist_ok=True)
        save_image(images[i], f"generated_images/GAN/{i}.png")

    print("Generated GAN images saved to 'generated_images/' directory.")

elif run["CGAN"]:

    from models import cgans

    images = cgans.generate_images(num_images=num_images, latent_dim=128)

    # Save generated images
    for i in range(len(images)):
        os.makedirs("generated_images", exist_ok=True)
        os.makedirs("generated_images/CGAN", exist_ok=True)
        save_image(images[i], f"generated_images/CGAN/{i}.png")

    print("Generated CGAN images saved to 'generated_images/' directory.")

elif run["Diffusion"]:

    from models import diffusion_models 

    best_model = diffusion_models.MyDDPM(diffusion_models.MyUNet(), n_steps=1000, device=device)
    best_model.load_state_dict(torch.load('./models/netG_CGAN.pth', map_location=device))
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