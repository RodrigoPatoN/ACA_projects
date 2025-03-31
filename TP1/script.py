from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import pandas as pd
from torchvision.transforms.functional import to_pil_image
from scipy.stats import skew, kurtosis
from sklearn.metrics import confusion_matrix
import itertools
import plotly.express as px
import plotly.graph_objects as go
import torch.nn.functional as F
import json
import random
from torch.utils.data import ConcatDataset
from PIL import Image
import sys
import os

import medmnist
from medmnist import INFO, Evaluator

tqdm().disable = True

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Running on MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

data_flag = 'dermamnist'
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

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

# Encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

print(train_dataset)
print("===================")
print(test_dataset)

train_dataset.montage(length=1)

train_dataset.montage(length=20)

train_labels = train_dataset.labels.squeeze()

fig = go.Figure()
fig.add_trace(go.Histogram(x=train_labels, name='train'))

# increase font size to 14
fig.update_layout(font=dict(size=16))

fig.update_traces(texttemplate='%{y}', textposition='outside')
fig.update_layout(plot_bgcolor='white')
fig.update_yaxes(range=[0, 5000])
fig.update_xaxes(title_text='Label')
fig.update_yaxes(title_text='Nº of samples')
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))


def random_undersampling(dataset, n_samples_per_class):

    all_labels = np.array(dataset.labels.squeeze())
    unique_labels = np.unique(all_labels)

    undersampled_imgs = []
    undersampled_labels = []

    # Undersample each class
    for label in unique_labels:
        label_indices = np.where(all_labels == label)[0]

        if len(label_indices) > n_samples_per_class:
            label_indices = np.random.choice(label_indices, n_samples_per_class, replace=False)
        
        imgs = [dataset[i][0] for i in label_indices]
        undersampled_imgs.extend(imgs)
        undersampled_labels.extend([label] * len(imgs))
    
    # Shuffle the undersampled dataset
    indices = np.random.permutation(len(undersampled_imgs))
    undersampled_imgs = torch.tensor(np.array(undersampled_imgs)[indices])
    undersampled_labels = torch.tensor(np.array(undersampled_labels)[indices])

    return list(zip(undersampled_imgs, undersampled_labels))


def augment_undersampled_dataset(dataset, n_samples_per_class=700):

    rotation = transforms.RandomRotation(30)
    translation = transforms.RandomAffine(0, translate=(0.1, 0.1))
    h_flip = transforms.RandomHorizontalFlip(p=1.0)  # Always apply
    v_flip = transforms.RandomVerticalFlip(p=1.0)  # Always apply

    augmented_imgs = []
    augmented_labels = []

    all_labels = []
    all_imgs = []
    
    for imgs, labels in dataset:
        all_imgs.append(imgs)
        all_labels.append(labels)

    all_labels = np.array(all_labels)
    unique_labels = np.unique(all_labels)

    for label in unique_labels:

        print(label)
        
        class_indices = np.where(all_labels == label)[0]
        class_imgs = [all_imgs[i] for i in class_indices]

        augmented_imgs_class = class_imgs.copy()


        while len(augmented_imgs_class) < n_samples_per_class:
            
            # Randomly select an image from the class
            img = random.choice(class_imgs)

            # Ensure img is a PIL image (if it's numpy, convert it)
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)  # Convert numpy array to PIL image

            # Randomly choose a transformation
            transformation = random.choice([rotation, translation, h_flip, v_flip])

            # Apply the transformation
            transformed_img = transformation(img)
            augmented_imgs_class.append(transformed_img)

        augmented_imgs.extend(augmented_imgs_class)
        augmented_labels.extend([label] * len(augmented_imgs_class))

    # Shuffle the augmented dataset
    indices = np.random.permutation(len(augmented_imgs))
    augmented_imgs = torch.tensor(np.array(augmented_imgs)[indices])
    augmented_labels = torch.tensor(np.array(augmented_labels)[indices])

    print(f"Augmented dataset size: {len(augmented_imgs)}")
    print(augmented_imgs[0].shape)

    return list(zip(augmented_imgs, augmented_labels))

undersampled_data = random_undersampling(train_dataset, 700)

labels = [label for _, label in undersampled_data]

fig = go.Figure()
fig.add_trace(go.Histogram(x=labels, name='undersampled'))

# increase font size to 14
fig.update_layout(font=dict(size=16))

fig.update_traces(texttemplate='%{y}', textposition='outside')
fig.update_layout(plot_bgcolor='white')
fig.update_yaxes(range=[0, 1100])
fig.update_xaxes(title_text='Label')
fig.update_yaxes(title_text='Nº of samples')
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

augmented_data = augment_undersampled_dataset(undersampled_data, 700)


labels = [label for _, label in augmented_data]

fig = go.Figure()
fig.add_trace(go.Histogram(x=labels, name='undersampled'))

# Stack input features
X_train = torch.stack([sample[0] for sample in augmented_data])
X_test = torch.stack([sample[0] for sample in test_dataset])

# Convert labels to 1D tensor
labels_train = [label for _,label in augmented_data]
y_train = torch.tensor(labels_train)
y_test = torch.tensor(test_dataset.labels.squeeze())

X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)


X_train_undersampled = torch.stack([sample[0] for sample in undersampled_data])
labels_train_undersampled = [label for _,label in undersampled_data]
y_train_undersampled = torch.tensor(labels_train_undersampled)

X_test_undersampled = torch.stack([sample[0] for sample in test_dataset])
y_test_undersampled = torch.tensor(test_dataset.labels.squeeze())

class DNN(nn.Module):

    def __init__(self, input_size, hidden_sizes, num_classes, activation):

        super(DNN, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        if len(hidden_sizes) == 0:
            self.layers.append(nn.Linear(input_size, num_classes))
        else:
            self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

            if activation == 'relu':
                self.activations.append(nn.ReLU())
            elif activation == 'sigmoid':
                self.activations.append(nn.Sigmoid())
            elif activation == 'tanh':
                self.activations.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation function: {activation}")
        
            for i in range(len(hidden_sizes)-1):

                self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))

                if activation == 'relu':
                    self.activations.append(nn.ReLU())
                elif activation == 'sigmoid':
                    self.activations.append(nn.Sigmoid())
                elif activation == 'tanh':
                    self.activations.append(nn.Tanh())
                else:
                    raise ValueError(f"Unknown activation function: {activation}")

            self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))
    
    def forward(self, x):

        for i in range(len(self.activations)):
            x = self.layers[i](x)  # Linear layer
            x = self.activations[i](x)  # Activation function
        
        x = self.layers[-1](x)
        
        return x
    

def fit(X_train, y_train, nn, criterion, optimizer, n_epochs, to_device=True, batch_size=32):
    
    #send everything to the device (ideally a GPU)
    if to_device:
        nn = nn.to(device)
        X_train = X_train.to(device)
        y_train = y_train.to(device).long()

    # Train the network
    loss_values = []
    for epoch in range(n_epochs):

        accu_loss = 0

        for i in range(0, X_train.size(0), batch_size):

            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward pass
            outputs = nn(X_batch)
            loss = criterion(outputs, y_batch)
            accu_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #if (epoch+1) % 10 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, accu_loss))
        loss_values.append(accu_loss)

    return loss_values, nn.to("cpu")


def evaluate_network(net, X, y, to_device=True):
    
    net.eval()
    if to_device:
        X = X.to(device)
        net = net.to(device)

    # Run the model on the test data
    with torch.no_grad():
        outputs = net(X)
        _, predicted = torch.max(outputs.data, 1)

    # Convert tensors to numpy arrays
    if to_device:
        predicted = predicted.to("cpu")
    predicted_np = predicted.numpy()
    test_target_np = y.numpy()

    # Compute confusion matrix and F1 score
    conf_mat = confusion_matrix(test_target_np, predicted_np)
    
    return conf_mat


parameters_test = {
    "n_epochs": [50],
    "batch_size": [32, 64],
    "learning_rate": [0.001, 0.01, 0.1],
    "n_layers": [3, 15],
    "activation_function": ['relu', 'sigmoid'],
    "loss_function": ["Cross Entropy", "Multi Margin"], #nn.CrossEntropyLoss(), nn.MultiMarginLoss()],
    "optimizer": ["ADAM", "SGD", "RMSprop"],
}

# Create a grid of hyperparameters
param_values = [v for v in parameters_test.values()]
param_names = [k for k in parameters_test.keys()]
param_combinations = list(itertools.product(*param_values))

results = {}
k = 5

#get the combinations which have already been tested
already_tested_dnn = pd.read_json("results_dnn.json")
already_tested_dnn = already_tested_dnn.transpose()

already_tested_params_combinations = already_tested_dnn[param_names].values
already_tested_params_combinations = [tuple(row) for row in already_tested_params_combinations]

param_combinations = [params for params in param_combinations if params not in already_tested_params_combinations]

# Loop over all hyperparameter combinations
for i, params in enumerate(param_combinations):

    print(f"Testing hyperparameter combination {i+1}/{len(param_combinations)}")
    print(params)
    param_dict = dict(zip(param_names, params))

    results[i] = {**param_dict, "results": []}

    # Unpack the parameters
    n_epochs, batch_size, learning_rate, n_layers, activation_function, loss_function, optimizer_name = params
    num_inputs = X_train_flattened.shape[1]

    hidden_layers_sizes = ((num_inputs + n_classes) // 2,) * n_layers
    total_data_train = X_train_flattened.shape[0]
    total_data_fold = total_data_train // k

    for fold in range(k):

        idxs_val_fold = list(range(fold*total_data_fold, (fold+1)*total_data_fold))
        idxs_train_fold = list(set(range(total_data_train)) - set(idxs_val_fold))

        X_val_flattened_fold = X_train_flattened[idxs_val_fold]
        y_val_fold = y_train[idxs_val_fold]

        X_train_flattened_fold = X_train_flattened[idxs_train_fold]
        y_train_fold = y_train[idxs_train_fold]

        # Define the network
        dnn = DNN(input_size = num_inputs,
                hidden_sizes = hidden_layers_sizes, 
                num_classes = n_classes, 
                activation = activation_function)

        # Define the loss function and optimizer
        if loss_function == "Cross Entropy":
            criterion = nn.CrossEntropyLoss()
        elif loss_function == "Multi Margin":
            criterion = nn.MultiMarginLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        
        if optimizer_name == "ADAM":
            optimizer = optim.Adam(dnn.parameters(), lr=learning_rate)
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(dnn.parameters(), lr=learning_rate)
        elif optimizer_name == "RMSprop":
            optimizer = optim.RMSprop(dnn.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Train the network
        loss_values, trained_net = fit(X_train_flattened_fold, 
                                    y_train_fold.long(), 
                                    dnn,
                                    criterion,
                                    optimizer,
                                    n_epochs = n_epochs, 
                                    batch_size = batch_size, 
                                    to_device = False)

        # Evaluate the network
        conf_mat_train = evaluate_network(trained_net, X_train_flattened_fold, y_train_fold)
        conf_mat = evaluate_network(trained_net, X_val_flattened_fold, y_val_fold)

        # Store the results
        results[i + len(already_tested_params_combinations)]["results"].append({
            'loss_values': loss_values,
            'confusion_matrix_train': conf_mat_train.tolist(),
            'confusion_matrix_val': conf_mat.tolist()
        })

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)


class CNN(nn.Module):

    def __init__(self, num_conv_layers, conv_out_channels,
                 conv_kernel_size, conv_padding, num_hidden_layers, 
                 activation_function, pooling, input_channels, num_classes):
        
        """
        Fully configurable CNN model
        
        Args:
        - input_channels (int): Number of input channels (e.g., 3 for RGB).
        - num_classes (int): Number of output classes.
        - num_conv_layers (int): Number of convolutional layers.
        - conv_out_channels (int): Number of output channels for each conv layer.
        - conv_kernel_size (int): Kernel size for convolutional layers.
        - conv_padding (int): Padding for convolutional layers.
        - num_hidden_layers (int): Number of fully connected hidden layers.
        - activation_function: Activation function
        - pooling (nn.Module): Pooling layer
        """

        super(CNN, self).__init__()

        self.activation = activation_function
        self.pooling = pooling

        # Create convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels

        for _ in range(num_conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, conv_out_channels, kernel_size=conv_kernel_size, padding=conv_padding))
            in_channels = conv_out_channels

        feature_size = self._compute_flattened_size(28, num_conv_layers, conv_kernel_size, conv_padding)

        self.fc_layers = nn.ModuleList()
        fc_input_size = feature_size

        # in here we are assuming all fully connected layers will have the same size - done in order to be consistent with dnn
        for _ in range(num_hidden_layers):
            self.fc_layers.append(nn.Linear(fc_input_size, fc_input_size))

        self.output_layer = nn.Linear(fc_input_size, num_classes)


    def _compute_flattened_size(self, input_size, num_conv_layers, kernel_size, padding):

        """ Helper function to compute the flattened size after convolutions """
        for _ in range(num_conv_layers):
            input_size = (input_size - kernel_size + 2 * padding) + 1  # Conv effect
            input_size //= 2  # Max pooling effect
        return input_size * input_size * self.conv_layers[-1].out_channels


    def forward(self, x):

        for conv in self.conv_layers:
            x = self.pooling(self.activation(conv(x)))
        
        x = x.view(x.shape[0], -1)

        for fc in self.fc_layers:
            x = self.activation(fc(x))
        
        x = self.output_layer(x)
        return x


parameters_test_cnn = {
    "n_epochs": [50],
    "activation_function": ['relu', 'sigmoid'],
    "pooling": ["MaxPool", "AvgPool"], #[nn.MaxPool2d(2), nn.AvgPool2d(2)],
    "n_conv_layers": [1, 2],
    "conv_out_channels": [16],
    "conv_kernel_size": [3],
    "conv_padding": [1],
    "n_layers": [3, 15],
    "batch_size": [32, 64],
    "learning_rate": [0.001, 0.01, 0.1],
    "loss_function": ["Cross Entropy", "Multi Margin"], #[nn.CrossEntropyLoss(), nn.MultiMarginLoss()],
    "optimizer": ["ADAM", "SGD", "RMSprop"],
}

param_values = [v for v in parameters_test_cnn.values()]
param_names = [k for k in parameters_test_cnn.keys()]
param_combinations = list(itertools.product(*param_values))

print(os.listdir("./"))

already_tested_cnn = pd.read_json("results_cnn_1.json").T

already_tested_params_combinations = already_tested_cnn[param_names].values
already_tested_params_combinations = [tuple(row) for row in already_tested_params_combinations]

param_combinations = [params for params in param_combinations if params not in already_tested_params_combinations]
total_combinations = len(param_combinations)

results = {}

for i, params in tqdm(enumerate(param_combinations), total=total_combinations, desc="Hyperparameter Search"):

    print(f"\nTesting combination {i+1}/{total_combinations}")
    param_dict = dict(zip(param_names, params))
    print(param_dict)

    results[i + len(already_tested_params_combinations)] = {**param_dict, "results": []}

    if param_dict["activation_function"] == "relu":
        activation = nn.ReLU()
    elif param_dict["activation_function"] == "sigmoid":
        activation = nn.Sigmoid()
    else:
        raise ValueError(f"Unknown activation function: {param_dict['activation_function']}")
    
    if param_dict["pooling"] == "MaxPool":
        pooling = nn.MaxPool2d(2)
    elif param_dict["pooling"] == "AvgPool":
        pooling = nn.AvgPool2d(2)
    else:
        raise ValueError(f"Unknown pooling layer: {param_dict['pooling']}")
    
    total_data_train = X_train.shape[0]
    total_data_fold = total_data_train // k

    for fold in range(k):

        idxs_val_fold = list(range(fold*total_data_fold, (fold+1)*total_data_fold))
        idxs_train_fold = list(set(range(total_data_train)) - set(idxs_val_fold))

        X_val_fold = X_train[idxs_val_fold]
        y_val_fold = y_train[idxs_val_fold]

        X_train_fold = X_train[idxs_train_fold]
        y_train_fold = y_train[idxs_train_fold]

        # Define the network
        cnn = CNN(
            input_channels = 3,
            num_classes = n_classes,
            num_conv_layers = param_dict["n_conv_layers"],
            conv_out_channels=param_dict["conv_out_channels"],
            conv_kernel_size=param_dict["conv_kernel_size"],
            conv_padding=param_dict["conv_padding"],
            activation_function=activation,
            pooling=pooling,
            num_hidden_layers=param_dict["n_layers"],
        )

        # Define loss function and optimizer

        if param_dict["loss_function"] == "Cross Entropy":
            criterion = nn.CrossEntropyLoss()
        elif param_dict["loss_function"] == "Multi Margin":
            criterion = nn.MultiMarginLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        
        optimizer = {
            "ADAM": optim.Adam(cnn.parameters(), lr=param_dict["learning_rate"]),
            "SGD": optim.SGD(cnn.parameters(), lr=param_dict["learning_rate"]),
            "RMSprop": optim.RMSprop(cnn.parameters(), lr=param_dict["learning_rate"]),
        }[param_dict["optimizer"]]

        # Train the network
        loss_values, trained_net = fit(X_train_fold,
                                    y_train_fold.long(), 
                                    cnn, 
                                    criterion, 
                                    optimizer, 
                                    param_dict["n_epochs"], 
                                    param_dict["batch_size"])

        # Evaluate the network
        conf_mat_train = evaluate_network(trained_net, X_train_fold, y_train_fold)
        conf_mat_val = evaluate_network(trained_net, X_val_fold, y_val_fold)

        # Store results
        results[i + len(already_tested_params_combinations)]["results"].append({
            'loss_values': loss_values,
            'confusion_matrix_train': conf_mat_train.tolist(),
            'confusion_matrix_val': conf_mat_val.tolist(),
        })

    with open("results_cnn.json", "w") as f:
        json.dump(results, f, indent=4)


def evaluate_loss(model, X_val, y_val, criterion):
    model.eval()  # Set to evaluation mode (disables dropout, batch norm updates)
    with torch.no_grad():  # No gradient computation for faster evaluation
        outputs = model(X_val)
        loss = criterion(outputs, y_val.long())  # Compute loss
    return loss.item()


# Now I will do the training with early stopping based on val loss - only the 50 best combinations will be tested

best_50_df = pd.read_csv("data_cnn_50_best.csv")
best_50_parmas = best_50_df[param_names].values
best_50_params = [tuple(row) for row in best_50_parmas]


best_50_params = []

patience = 5  # Number of epochs to wait for improvement
min_delta = 1e-4  # Minimum change in loss to qualify as an improvement

results = {}

for i, params in tqdm(enumerate(best_50_params), total=50, desc="Hyperparameter Search"):

    break

    print(f"\nTesting combination {i+1}/{50}")
    param_dict = dict(zip(param_names, params))
    print(param_dict)

    results[i + len(already_tested_params_combinations)] = {**param_dict, "results": []}

    activation = nn.ReLU() if param_dict["activation_function"] == "relu" else nn.Sigmoid()
    pooling = nn.MaxPool2d(2) if param_dict["pooling"] == "MaxPool" else nn.AvgPool2d(2)

    total_data_train = X_train.shape[0]
    total_data_fold = total_data_train // k

    for fold in range(k):

        idxs_val_fold = list(range(fold*total_data_fold, (fold+1)*total_data_fold))
        idxs_train_fold = list(set(range(total_data_train)) - set(idxs_val_fold))

        X_val_fold, y_val_fold = X_train[idxs_val_fold], y_train[idxs_val_fold]
        X_train_fold, y_train_fold = X_train[idxs_train_fold], y_train[idxs_train_fold]

        cnn = CNN(
            input_channels=3,
            num_classes=n_classes,
            num_conv_layers=param_dict["n_conv_layers"],
            conv_out_channels=param_dict["conv_out_channels"],
            conv_kernel_size=param_dict["conv_kernel_size"],
            conv_padding=param_dict["conv_padding"],
            activation_function=activation,
            pooling=pooling,
            num_hidden_layers=param_dict["n_layers"],
        )

        criterion = nn.CrossEntropyLoss() if param_dict["loss_function"] == "Cross Entropy" else nn.MultiMarginLoss()
        optimizer = {
            "ADAM": optim.Adam(cnn.parameters(), lr=param_dict["learning_rate"]),
            "SGD": optim.SGD(cnn.parameters(), lr=param_dict["learning_rate"]),
            "RMSprop": optim.RMSprop(cnn.parameters(), lr=param_dict["learning_rate"]),
        }[param_dict["optimizer"]]

        # Early Stopping Initialization
        best_val_loss = np.inf
        epochs_no_improve = 0
        best_model = None
        loss_values = []

        for epoch in range(param_dict["n_epochs"]):
            train_loss, trained_net = fit(X_train_fold, y_train_fold.long(), cnn, criterion, optimizer, 1, param_dict["batch_size"])
            val_loss = evaluate_loss(trained_net, X_val_fold, y_val_fold, criterion)

            loss_values.append(train_loss)
            print(f"Epoch {epoch+1}: Train Loss = {train_loss[0]:.4f}, Val Loss = {val_loss:.4f}")

            # Check if validation loss improved
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model = trained_net
            else:
                epochs_no_improve += 1

            # Stop if no improvement for `patience` epochs
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Evaluate the best saved model
        conf_mat_train = evaluate_network(best_model, X_train_fold, y_train_fold)
        conf_mat_val = evaluate_network(best_model, X_val_fold, y_val_fold)

        results[i + len(already_tested_params_combinations)]["results"].append({
            'loss_values': loss_values,
            'confusion_matrix_train': conf_mat_train.tolist(),
            'confusion_matrix_val': conf_mat_val.tolist(),
        })

    with open("results_cnn.json", "w") as f:
        json.dump(results, f, indent=4)


for i, params in tqdm(enumerate(best_50_params), total=50, desc="Hyperparameter Search"):

    print(f"\nTesting combination {i+1}/{total_combinations}")
    param_dict = dict(zip(param_names, params))
    print(param_dict)

    results[i + len(already_tested_params_combinations)] = {**param_dict, "results": []}

    if param_dict["activation_function"] == "relu":
        activation = nn.ReLU()
    elif param_dict["activation_function"] == "sigmoid":
        activation = nn.Sigmoid()
    else:
        raise ValueError(f"Unknown activation function: {param_dict['activation_function']}")
    
    if param_dict["pooling"] == "MaxPool":
        pooling = nn.MaxPool2d(2)
    elif param_dict["pooling"] == "AvgPool":
        pooling = nn.AvgPool2d(2)
    else:
        raise ValueError(f"Unknown pooling layer: {param_dict['pooling']}")
    
    total_data_train = X_train_undersampled.shape[0]
    total_data_fold = total_data_train // k

    for fold in range(k):

        idxs_val_fold = list(range(fold*total_data_fold, (fold+1)*total_data_fold))
        idxs_train_fold = list(set(range(total_data_train)) - set(idxs_val_fold))

        X_val_fold = X_train_undersampled[idxs_val_fold]
        y_val_fold = y_train_undersampled[idxs_val_fold]

        X_train_fold = X_train_undersampled[idxs_train_fold]
        y_train_fold = y_train_undersampled[idxs_train_fold]

        # Define the network
        cnn = CNN(
            input_channels = 3,
            num_classes = n_classes,
            num_conv_layers = param_dict["n_conv_layers"],
            conv_out_channels=param_dict["conv_out_channels"],
            conv_kernel_size=param_dict["conv_kernel_size"],
            conv_padding=param_dict["conv_padding"],
            activation_function=activation,
            pooling=pooling,
            num_hidden_layers=param_dict["n_layers"],
        )

        # Define loss function and optimizer

        if param_dict["loss_function"] == "Cross Entropy":
            criterion = nn.CrossEntropyLoss()
        elif param_dict["loss_function"] == "Multi Margin":
            criterion = nn.MultiMarginLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        
        optimizer = {
            "ADAM": optim.Adam(cnn.parameters(), lr=param_dict["learning_rate"]),
            "SGD": optim.SGD(cnn.parameters(), lr=param_dict["learning_rate"]),
            "RMSprop": optim.RMSprop(cnn.parameters(), lr=param_dict["learning_rate"]),
        }[param_dict["optimizer"]]

        # Train the network
        loss_values, trained_net = fit(X_train_fold,
                                    y_train_fold.long(), 
                                    cnn, 
                                    criterion, 
                                    optimizer, 
                                    param_dict["n_epochs"], 
                                    param_dict["batch_size"])

        # Evaluate the network
        conf_mat_train = evaluate_network(trained_net, X_train_fold, y_train_fold)
        conf_mat_val = evaluate_network(trained_net, X_val_fold, y_val_fold)

        # Store results
        results[i + len(already_tested_params_combinations)]["results"].append({
            'loss_values': loss_values,
            'confusion_matrix_train': conf_mat_train.tolist(),
            'confusion_matrix_val': conf_mat_val.tolist(),
        })

    with open("results_cnn.json", "w") as f:
        json.dump(results, f, indent=4)