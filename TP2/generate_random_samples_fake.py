import torch
import os
from torchvision.utils import save_image
import numpy as np

run = {
    "VAE": True,
    "DAE": False,
    "GAN": False,
    "CGAN": False,
    "Diffusion": False,
}

num_images = 10000
SEEDS = [42, 123, 2024, 7, 888]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


os.makedirs("generated_images", exist_ok=True)

for seed_num, seed in enumerate(SEEDS):

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Using seed {seed} for random sampling.")

    if run["VAE"]:

        from models import autoencoders 

        vae_01 = autoencoders.VAE(color_channels=3, latent_dim=128)
        vae_01.load_state_dict(torch.load(f"trained_models/{seed}/vae_model_0.01.pth"))

        vae_001 = autoencoders.VAE(color_channels=3, latent_dim=128)
        vae_001.load_state_dict(torch.load(f"trained_models/{seed}/vae_model_0.001.pth"))

        os.makedirs("generated_images/VAE", exist_ok=True)
        os.makedirs(f"generated_images/VAE/{seed_num}", exist_ok=True)
        os.makedirs(f"generated_images/VAE/{seed_num}/001", exist_ok=True)
        os.makedirs(f"generated_images/VAE/{seed_num}/01", exist_ok=True)

        generated_01 = autoencoders.generate_images_vae(vae_01, num_images=num_images, latent_dim=128)
        generated_001 = autoencoders.generate_images_vae(vae_001, num_images=num_images, latent_dim=128)

        for i in range(len(generated_01)):
            save_image(generated_01[i], f"generated_images/VAE/{seed_num}/01/{i}.png")
            save_image(generated_001[i], f"generated_images/VAE/{seed_num}/001/{i}.png")

        print(f"Generated AE images for seed {seed}")

    elif run["DAE"]:
            
        from models import autoencoders 

        dae_01 = autoencoders.DenoisingAutoencoder(color_channels=3, latent_dim=128)
        dae_01.load_state_dict(torch.load(f"trained_models/{seed}/dae_models_0.01.pth"))

        dae_001 = autoencoders.DenoisingAutoencoder(color_channels=3, latent_dim=128)
        dae_001.load_state_dict(torch.load(f"trained_models/{seed}/dae_models_0.001.pth"))

        os.makedirs("generated_images/DAE", exist_ok=True)
        os.makedirs(f"generated_images/DAE/{seed_num}", exist_ok=True)
        os.makedirs(f"generated_images/DAE/{seed_num}/001", exist_ok=True)
        os.makedirs(f"generated_images/DAE/{seed_num}/01", exist_ok=True)

        generated_01 = autoencoders.generate_images_dae(dae_01, num_images=num_images, latent_dim=128, random_seed=seed)
        generated_001 = autoencoders.generate_images_dae(dae_001, num_images=num_images, latent_dim=128, random_seed=seed)

        for i in range(len(generated_01)):
            save_image(generated_01[i], f"generated_images/DAE/{seed_num}/01/{i}.png")
            save_image(generated_001[i], f"generated_images/DAE/{seed_num}/001/{i}.png")

    elif run["GAN"]:

        from models import gans

        generator_01 = gans.Generator()
        generator_01.load_state_dict(torch.load(f"trained_models/{seed}/GAN_netG_0.01.pth", map_location=device))
        generator_01.to(device)

        generator_001 = gans.Generator()
        generator_001.load_state_dict(torch.load(f"trained_models/{seed}/GAN_netG_0.001.pth", map_location=device))
        generator_001.to(device)

        images_01 = gans.generate_images(generator_01, num_images=num_images, device=device)
        images_001 = gans.generate_images(generator_001, num_images=num_images, device=device)

        os.makedirs("generated_images/GAN", exist_ok=True)
        os.makedirs(f"generated_images/GAN/{seed_num}", exist_ok=True)
        os.makedirs(f"generated_images/GAN/{seed_num}/001", exist_ok=True)
        os.makedirs(f"generated_images/GAN/{seed_num}/01", exist_ok=True)

        # Save generated images
        for i in range(len(images_01)):
            save_image(images_01[i], f"generated_images/GAN/{seed_num}/01/{i}.png")
            save_image(images_001[i], f"generated_images/GAN/{seed_num}/001/{i}.png")

        print(f"Generated GAN images for seed {seed}")

    elif run["CGAN"]:

        from models import cgans

        generator_01 = cgans.Generator(num_classes=8)
        generator_01.load_state_dict(torch.load(f"trained_models/{seed}/CGAN_netG_0.01.pth", map_location=device))
        generator_01.to(device)

        generator_001 = cgans.Generator(num_classes=8)
        generator_001.load_state_dict(torch.load(f"trained_models/{seed}/CGAN_netG_0.001.pth", map_location=device))
        generator_001.to(device)

        images_01 = cgans.generate_images(generator_01, num_images=num_images, device=device)
        images_001 = cgans.generate_images(generator_001, num_images=num_images, device=device)

        os.makedirs("generated_images/CGAN", exist_ok=True)
        os.makedirs(f"generated_images/CGAN/{seed_num}", exist_ok=True)
        os.makedirs(f"generated_images/CGAN/{seed_num}/001", exist_ok=True)
        os.makedirs(f"generated_images/CGAN/{seed_num}/01", exist_ok=True)

        # Save generated images
        for i in range(len(images_01)):
            save_image(images_01[i], f"generated_images/CGAN/{seed_num}/01/{i}.png")
            save_image(images_001[i], f"generated_images/CGAN/{seed_num}/001/{i}.png")

        print("Generated CGAN images saved to 'generated_images/' directory.")

    elif run["Diffusion"]:

        from models import diffusion_models 

        best_model_01 = diffusion_models.MyDDPM(diffusion_models.MyUNet(), n_steps=1000, device=device)
        best_model_01.load_state_dict(torch.load(f'./trained_models/{seed}/diff_model_0.01.pt', map_location=device))
        best_model_01.eval()

        best_model_001 = diffusion_models.MyDDPM(diffusion_models.MyUNet(), n_steps=1000, device=device)
        best_model_001.load_state_dict(torch.load(f'./trained_models/{seed}/diff_model_0.001.pt', map_location=device))
        best_model_001.eval()

        generated_01 = diffusion_models.generate_new_images(
                best_model_01,
                option = 1,
                n_samples=16,# change the number of samples as needed
                device=device,
                gif_name="test.gif"
            )
        
        generated_001 = diffusion_models.generate_new_images(
                best_model_001,
                option = 1,
                n_samples=16,# change the number of samples as needed
                device=device,
                gif_name="test.gif"
            )
        
        #diffusion_models.show_images(generated, "Final Option 1 result")