#!/usr/bin/env python
# coding: utf-8

# In[11]:


#!pip install medmnist


# In[ ]:


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

import medmnist
from medmnist import INFO, Evaluator


# In[4]:


tqdm().disable = True


# In[5]:


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Running on MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")


# In[6]:


print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")


# # We first work on a 2D dataset with size 28x28

# In[7]:


data_flag = 'dermamnist'
download = True
BATCH_SIZE = 128

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])


# ## First, we read the MedMNIST data, preprocess them and encapsulate them into dataloader form.

# In[8]:


# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
val_dataset = DataClass(split='val', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

pil_dataset = DataClass(split='train', download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)


# In[9]:


print(train_dataset)
print("===================")
print(val_dataset)
print("===================")
print(test_dataset)


# In[10]:


# visualization

train_dataset.montage(length=1)


# In[11]:


# montage

train_dataset.montage(length=20)


# Stack input features
X_train = torch.stack([sample[0] for sample in train_dataset])
X_val = torch.stack([sample[0] for sample in val_dataset])
X_test = torch.stack([sample[0] for sample in test_dataset])

# Convert labels to 1D tensor
y_train = torch.tensor(train_dataset.labels.squeeze())
y_val = torch.tensor(val_dataset.labels.squeeze())
y_test = torch.tensor(test_dataset.labels.squeeze())

X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_val_flattened = X_val.reshape(X_val.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)


# #### Get the MLPs

# In[14]:


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
    # Set the model to evaluation mode
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


# In[15]:


parameters_test = {
    "n_epochs": [50],
    "batch_size": [32, 64],
    "learning_rate": [0.001, 0.01, 0.1],
    "n_layers": [3, 15],
    "activation_function": ['relu', 'sigmoid'],
    "loss_function": [nn.CrossEntropyLoss(), nn.MultiMarginLoss()],
    "optimizer": ["ADAM", "SGD", "RMSprop"],
}

# Create a grid of hyperparameters
param_values = [v for v in parameters_test.values()]
param_names = [k for k in parameters_test.keys()]
param_combinations = list(itertools.product(*param_values))

results = {}

# Loop over all hyperparameter combinations

"""
for i, params in enumerate(param_combinations):
    print(f"Testing hyperparameter combination {i+1}/{len(param_combinations)}")
    print(params)

    # Unpack the parameters
    n_epochs, batch_size, learning_rate, n_layers, activation_function, loss_function, optimizer_name = params
    num_inputs = X_train_flattened.shape[1]

    hidden_layers_sizes = ((num_inputs + n_classes) // 2,) * n_layers
    print(hidden_layers_sizes)

    # Define the network
    dnn = DNN(input_size = num_inputs,
            hidden_sizes = hidden_layers_sizes, 
            num_classes = n_classes, 
            activation = activation_function)

    # Define the loss function and optimizer
    criterion = loss_function
    if optimizer_name == "ADAM":
        optimizer = optim.Adam(dnn.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(dnn.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(dnn.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Train the network
    loss_values, trained_net = fit(X_train_flattened, 
                                y_train, 
                                dnn,
                                criterion,
                                optimizer,
                                n_epochs = n_epochs, 
                                batch_size = batch_size, 
                                to_device = False)

    # Evaluate the network
    conf_mat_train = evaluate_network(trained_net, X_train_flattened, y_train)
    conf_mat = evaluate_network(trained_net, X_val_flattened, y_val)

    # Store the results
    results[params] = {
        'loss_values': loss_values,
        'confusion_matrix': conf_mat
    }

    # results saved to disk after each iteration to ensure that we don't lose everything if the code crashes
    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv")
"""

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
    "activation_function": ['relu', 'sigmoid'],
    "pooling": [nn.MaxPool2d(2), nn.AvgPool2d(2)],
    "n_conv_layers": [1, 2],
    "conv_out_channels": [16],
    "conv_kernel_size": [3],
    "conv_padding": [1],
    "n_layers": [3, 15],
    "batch_size": [32, 64],
    "learning_rate": [0.001, 0.01, 0.1],
    "n_epochs": [50],
    "loss_function": [nn.CrossEntropyLoss(), nn.MultiMarginLoss()],
    "optimizer": ["ADAM", "SGD", "RMSprop"],
}

# Create a grid of hyperparameters
param_values = [v for v in parameters_test_cnn.values()]
param_names = [k for k in parameters_test_cnn.keys()]
param_combinations = list(itertools.product(*param_values))
total_combinations = len(param_combinations)

for i, params in tqdm(enumerate(param_combinations), total=total_combinations, desc="Hyperparameter Search"):

    print(f"\nTesting combination {i+1}/{total_combinations}")
    
    param_dict = dict(zip(param_names, params))
    print(param_dict)

    if param_dict["activation_function"] == "relu":
        activation = nn.ReLU()
    elif param_dict["activation_function"] == "sigmoid":
        activation = nn.Sigmoid()
    else:
        raise ValueError(f"Unknown activation function: {param_dict['activation_function']}")

    # Define the network
    cnn = CNN(
        input_channels = 3,
        num_classes = n_classes,
        num_conv_layers = param_dict["n_conv_layers"],
        conv_out_channels=param_dict["conv_out_channels"],
        conv_kernel_size=param_dict["conv_kernel_size"],
        conv_padding=param_dict["conv_padding"],
        activation_function=activation,
        pooling=param_dict["pooling"],
        num_hidden_layers=param_dict["n_layers"],
    )

    # Define loss function and optimizer
    criterion = param_dict["loss_function"]
    optimizer = {
        "ADAM": optim.Adam(cnn.parameters(), lr=param_dict["learning_rate"]),
        "SGD": optim.SGD(cnn.parameters(), lr=param_dict["learning_rate"]),
        "RMSprop": optim.RMSprop(cnn.parameters(), lr=param_dict["learning_rate"]),
    }[param_dict["optimizer"]]

    # Train the network
    loss_values, trained_net = fit(X_train, y_train, cnn, criterion, optimizer, param_dict["n_epochs"], param_dict["batch_size"])

    # Evaluate the network
    conf_mat_train = evaluate_network(trained_net, X_train, y_train)
    conf_mat_val = evaluate_network(trained_net, X_val, y_val)

    # Store results
    results[str(param_dict)] = {
        'loss_values': loss_values,
        'confusion_matrix_train': conf_mat_train.tolist(),
        'confusion_matrix_val': conf_mat_val.tolist()
    }

    # Save results to JSON after each iteration
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Results saved!")
    