import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision.utils import save_image

import medmnist
from medmnist import INFO

# -------------------------------- Load Dataset --------------------------------

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

data_flag = 'bloodmnist'
download = True
BATCH_SIZE = 1

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

# -------------------------------- Setup --------------------------------   

# set random seed for reproducibility
SEEDS = [42, 123, 2024, 7, 888]
num_samples = 10000
i = 0

for seed in SEEDS:
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    i += 1

    # sample 10000 random samples for the dataset
    indices = np.random.choice(len(full_dataset), num_samples, replace=False)
    sampled_dataset = torch.utils.data.Subset(full_dataset, indices)
    sampled_dataloader = DataLoader(sampled_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Sampled {len(sampled_dataset)} images from the dataset with seed {seed}.")

    os.makedirs("sampled_dataset", exist_ok=True)
    os.makedirs(f"sampled_dataset/sample_{i}", exist_ok=True)

    # save the sampled dataset
    for j, (images, labels) in enumerate(sampled_dataloader):
        os.makedirs(f"sampled_dataset/sample_{i}", exist_ok=True)
        save_image(images, f"sampled_dataset/sample_{i}/{j}.png")
