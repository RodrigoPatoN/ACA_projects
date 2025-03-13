import json
import os
import itertools
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import medmnist
from medmnist import INFO

# Set device (MPS for Mac, CUDA for GPU, CPU as fallback)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Running on MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")

# Dataset parameters
data_flag = 'dermamnist'
download = True
BATCH_SIZE = 128

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# Preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
val_dataset = DataClass(split='val', transform=data_transform, download=download)

# Flatten inputs
X_train = torch.stack([sample[0] for sample in train_dataset])
X_val = torch.stack([sample[0] for sample in val_dataset])

X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_val_flattened = X_val.reshape(X_val.shape[0], -1)

y_train = torch.tensor(train_dataset.labels.squeeze())
y_val = torch.tensor(val_dataset.labels.squeeze())

# MLP Model
class DNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, activation):
        super(DNN, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        if len(hidden_sizes) == 0:
            self.layers.append(nn.Linear(input_size, num_classes))
        else:
            self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
            self.activations.append(self.get_activation(activation))
            
            for i in range(len(hidden_sizes) - 1):
                self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                self.activations.append(self.get_activation(activation))
            
            self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, x):
        for i in range(len(self.activations)):
            x = self.layers[i](x)
            x = self.activations[i](x)
        x = self.layers[-1](x)
        return x

# Training function
def fit(X_train, y_train, model, criterion, optimizer, n_epochs, batch_size):
    model = model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    
    loss_values = []
    for epoch in range(n_epochs):
        accu_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            accu_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {accu_loss:.4f}')
        loss_values.append(accu_loss)

    return loss_values, model.to("cpu")

# Evaluation function
def evaluate_network(model, X, y):
    model.eval()
    X = X.to(device)
    
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)

    predicted = predicted.to("cpu")
    conf_mat = confusion_matrix(y.numpy(), predicted.numpy())
    
    return conf_mat

# Hyperparameter search space
parameters_test = {
    "n_epochs": [50],
    "batch_size": [32, 64],
    "learning_rate": [0.001, 0.01, 0.1],
    "n_layers": [3, 15],
    "activation_function": ['relu', 'sigmoid'],
    "loss_function": [nn.CrossEntropyLoss(), nn.MultiMarginLoss()],
    "optimizer": ["Adam", "SGD"]
}

# Generate parameter combinations
param_combinations = list(itertools.product(*parameters_test.values()))
RESULTS_FILE = "grid_search_results.json"

# Function to run a single experiment
def run_experiment(params):
    n_epochs, batch_size, learning_rate, n_layers, activation_function, loss_function, optimizer_name = params
    num_inputs = X_train_flattened.shape[1]
    hidden_layers_sizes = ((num_inputs + n_classes) // 2,) * n_layers

    # Define the network
    dnn = DNN(input_size=num_inputs, hidden_sizes=hidden_layers_sizes, num_classes=n_classes, activation=activation_function)

    # Define loss function and optimizer
    criterion = loss_function

    if optimizer_name == "SGD":
        optimizer = optim.SGD(dnn.parameters(), lr=learning_rate)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(dnn.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Train the network
    loss_values, trained_net = fit(X_train_flattened, y_train, dnn, criterion, optimizer, n_epochs=n_epochs, batch_size=batch_size)

    # Evaluate the network
    conf_mat = evaluate_network(trained_net, X_val_flattened, y_val)

    # Format results
    result = {
        "params": {
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "n_layers": n_layers,
            "activation_function": activation_function,
            "loss_function": str(loss_function),
            "optimizer": optimizer_name
        },
        "loss_values": loss_values,
        "confusion_matrix": conf_mat.tolist()
    }

    # Save results to JSON
    save_results_to_json(result)

    return result

# Function to append results to JSON file
def save_results_to_json(new_result):
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)
    else:
        results = []

    results.append(new_result)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

# Run experiments in parallel
num_cores = -1  # Use all available CPU cores
results = Parallel(n_jobs=num_cores)(delayed(run_experiment)(params) for params in param_combinations)
