from time import perf_counter
import sys, os, matplotlib.pyplot as plt, numpy as np, pandas as pd
import torch,sys
import numpy as np
from torch import nn
from torch import optim
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

times_list_categorical_cpu, times_list_categorical_cuda, times_list_normal_cpu, times_list_normal_cuda = [], [], [], []

RANDOM_STATE = 100

CAT_EPOCHS = 3000
EPOCHS = 1500
NUM_LOOPS = 10
TRAIN_PERCENT : int = 80
LEARN_RATE = 0.01
LEARN_RATE_CAT = 0.001

def train_model(model : nn.Module, criterion, optimizer : optim, x_train : torch.Tensor, y_train : torch.Tensor,
                epochs : int = None, NUM_SUBSECTIONS : int = 0, LEARN_RATE : float = 0.1):
    start = perf_counter()
    
    if NUM_SUBSECTIONS:
        for i in range(NUM_SUBSECTIONS):
            model[i].train()
        for epoch in range(epochs):
            for i in  range(NUM_SUBSECTIONS):
                outputs = model[i](x_train[i])
                loss = criterion(outputs, y_train[i])
                # Backward pass and optimization
                optimizer[i].zero_grad()
                loss.backward()
                optimizer[i].step()
    else:
        model.train()
        for epoch in range(epochs):
            # training mode
            
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(f"time to train: {perf_counter() - start: .4f} (using {next(model.parameters()).device}, EPOCHS -- {epochs}, LEARN RATE -- {LEARN_RATE})")

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
    
    def forward(self, X : torch.Tensor) -> torch.Tensor:
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))
        X = self.fc3(X)
        return X
    
class LinearRegressionModelCat(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModelCat, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, X : torch.Tensor) -> torch.Tensor:
        return self.linear(X)

def main(device: str = 'cpu', cat : bool = None, LEARN_RATE : int = None, EPOCHS : int = 1, NUM_SUBSECTIONS : int = 0):
    # Load data remotely
    if cat:
        df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/iris.csv')
        y = df['Species']
        x = df.drop('Species', axis=1)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    else:
        df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
        y = df['logS']
        x = df.drop('logS', axis=1)
        
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x if cat else x_scaled, y, train_size=TRAIN_PERCENT / 100, random_state=RANDOM_STATE)
    # Convert data to PyTorch tensors
    x_train_tensor = torch.tensor(x_train.values if cat else x_train, dtype=torch.float32, device=device)
    x_test_tensor = torch.tensor(x_test.values if cat else x_test, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train if cat else y_train.values, dtype=torch.long if cat else torch.float, device=device)
    y_test_tensor = torch.tensor(y_test if cat else y_test.values, dtype=torch.long if cat else torch.float, device=device)
    
    if cat == False:
        y_train_tensor = y_train_tensor.view(-1, 1).to(device)
        y_test_tensor = y_test_tensor.view(-1, 1).to(device)

    if NUM_SUBSECTIONS > 1:
        x_train_subsets, y_train_subsets= [], []
        previous = 0
        for i in range (NUM_SUBSECTIONS):
            x_train_subsets.append(x_train_tensor[previous : previous + size])
            y_train_subsets.append(y_train_tensor[previous : previous + size])
            previous += size

    # Initialize model, loss function, and optimizer
    input_dim = x_train.shape[1]
    output_dim = len(label_encoder.classes_) if cat else 1
    model = LinearRegressionModelCat(input_dim, output_dim).to(device=device) if cat else LinearRegressionModel(input_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss() if cat else nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARN_RATE) if cat else optim.Adam(model.parameters(), lr=LEARN_RATE)
    train_model(model=model, criterion=criterion, optimizer=optimizer, x_train=x_train_tensor, y_train=y_train_tensor, epochs=EPOCHS, LEARN_RATE=LEARN_RATE, NUM_SUBSECTIONS=NUM_SUBSECTIONS)

    # y_lr_train_pred = predict(model, x_train_tensor)
    # y_lr_test_pred = predict(model, x_test_tensor)

    # y_lr_train_pred_np = y_lr_train_pred.cpu().numpy()
    # y_lr_test_pred_np = y_lr_test_pred.cpu().numpy()


for i in range(100, CAT_EPOCHS, CAT_EPOCHS // NUM_LOOPS):
    start = perf_counter()
    main(cat=True, LEARN_RATE=LEARN_RATE_CAT, EPOCHS = i)
    times_list_categorical_cpu.append(perf_counter() - start)
for i in range(100, CAT_EPOCHS, CAT_EPOCHS // NUM_LOOPS):
    start = perf_counter()
    main(cat=True, LEARN_RATE=LEARN_RATE_CAT, EPOCHS = i, device='cuda')
    times_list_categorical_cuda.append(perf_counter() - start)
    
    
for i in range(100, EPOCHS, EPOCHS // NUM_LOOPS):
    start = perf_counter()
    main(cat=False, LEARN_RATE=LEARN_RATE, EPOCHS = i)
    times_list_normal_cpu.append(perf_counter() - start)
for i in range(100, EPOCHS, EPOCHS // NUM_LOOPS):
    start = perf_counter()
    main(cat=False, LEARN_RATE=LEARN_RATE, EPOCHS = i, device='cuda')
    times_list_normal_cuda.append(perf_counter() - start)

# Plotting
# Calculate speedup for categorical data
speedup_list_categorical = [cpu_time / cuda_time for cpu_time, cuda_time in zip(times_list_categorical_cpu, times_list_categorical_cuda)]

# Calculate speedup for normal data
speedup_list_normal = [cpu_time / cuda_time for cpu_time, cuda_time in zip(times_list_normal_cpu, times_list_normal_cuda)]

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Plot categorical speedup
axes[0].plot(range(100, CAT_EPOCHS, CAT_EPOCHS // NUM_LOOPS), speedup_list_categorical, label='Categorical Speedup')
axes[0].set_title('Categorical Speedup')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Speedup')
axes[0].legend()
axes[0].grid(True)

# Plot normal speedup
axes[1].plot(range(100, EPOCHS, EPOCHS // NUM_LOOPS), speedup_list_normal, label='Normal Speedup')
axes[1].set_title('Normal Speedup')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Speedup')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
