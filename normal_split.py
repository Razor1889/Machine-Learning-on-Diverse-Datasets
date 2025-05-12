# imports
# NOTE -- made for windows

# run with -s -p for verbose outputs
import torch, matplotlib
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import sys, psutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.utils.parametrize as P
from time import perf_counter
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# local file
from master import my_print, GRAPH_VISIBILITY_ENABLED, LEARN_RATE, EPOCHS,\
    PRINTING_ENABLED, TRAIN_PERCENT, plot_predictions, SAVE_DIR, RANDOM_STATE, seed,\
        check_print, check_show_table, NUM_SUBSECTIONS
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Function to determine device based on command line argument
def get_device() -> str:
    if len(sys.argv) > 1 and sys.argv[1] == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

# function to save model for loading
def save_model(save_path : str = SAVE_DIR, model_name : str = None, model_to_save : torch.nn.Module = None) -> str:
    # save the model
    if not (model_name and model_to_save):
        return
    MODEL_PATH : pahthlib.WindowsPath = Path(save_path)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME : str = model_name
    MODEL_SAVE_PATH :pathlib.WindowsPath = MODEL_PATH / MODEL_NAME
    print(f"Saving model file to: {MODEL_SAVE_PATH}")
    torch.save(obj=model_to_save.state_dict(), f=MODEL_SAVE_PATH)
    return str(MODEL_SAVE_PATH)

# Linear Regression model
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
    
def train_model(model : nn.Module, criterion, optimizer : optim, x_train : torch.Tensor, y_train : torch.Tensor,
                epochs : int = EPOCHS, print_num : int = 20, enable_print : bool = False):
    if print_num > epochs:
        print_num = epochs
    start = perf_counter()
    schedulers = []
    for i in range(NUM_SUBSECTIONS):
        schedulers.append(ReduceLROnPlateau(optimizer[i], mode='min', factor=0.3, patience=20))
        model[i].train()
    for epoch in range(epochs):
        for i in range(NUM_SUBSECTIONS):
            # training mode
            y_pred = model[i](x_train[i])
            loss = criterion(y_pred, y_train[i])
            optimizer[i].zero_grad()
            loss.backward()
            optimizer[i].step()
            val_loss = criterion(model[i](x_train[i]), y_train[i])
            schedulers[i].step(val_loss)
        
        if (epoch + 1) % (epochs // print_num) == 0 and enable_print:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
    print(f"time to train: {perf_counter() - start: .4f} (using {next(model[0].parameters()).device}, EPOCHS -- {epochs}, LEARN RATE -- {LEARN_RATE})")

# Predictions
def predict(model : nn.Module, test_data : torch.Tensor):
    model.eval()
    with torch.inference_mode():
        outputs = model(test_data)
    return outputs

def get_epochs():
    if len(sys.argv) > 1:
        return int(sys.argv[len(sys.argv) - 1])
    return EPOCHS

def main() -> int:
    # Load data remotely
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')

    device = get_device()

    # Split data into features (x) and target (y)
    y = df['logS']
    x = df.drop('logS', axis=1)
    # print(df)
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, train_size=TRAIN_PERCENT / 100, random_state=RANDOM_STATE)
    # print(y_test.values, y_test.shape)
    # Convert data to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)
    # print([x.size() for x in (x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor)])

    size : int = len(x_train) // NUM_SUBSECTIONS
    x_train_subsets, y_train_subsets= [], []
    previous = 0
    for i in range (NUM_SUBSECTIONS):
        x_train_subsets.append(x_train_tensor[previous : previous + size])
        y_train_subsets.append(y_train_tensor[previous : previous + size])
        previous += size
    
    # Initialize model, loss function, and optimizer
    
    seed()
    models : list[torch.Module] = []
    optimizers : optim = []
    for i in range (NUM_SUBSECTIONS):
        input_dim = x_train_subsets[i].shape[1]    
        models.append(model := LinearRegressionModel(input_dim, 1).to(device=device))
        optimizers.append(optim.Adam(model.parameters(), lr=LEARN_RATE))
        
    criterion = nn.MSELoss()
    # Training the model
    seed()
    PRINTING_ENABLED : bool = check_print()
    SHOW_TABLE : bool = check_show_table()
    EPOCHS = get_epochs()
    if SHOW_TABLE:
        if device == 'cpu':
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
                with record_function("model_inference"):
                    train_model(models, criterion, optimizers, x_train_subsets, y_train_subsets, enable_print=PRINTING_ENABLED, epochs=EPOCHS)
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # in MB
                    print(f"CPU memory usage: {memory_usage} MB")
                    
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        else:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
                with record_function("model_inference"):
                    with P.cached():
                        train_model(models, criterion, optimizers, x_train_subsets, y_train_subsets, enable_print=PRINTING_ENABLED, epochs=EPOCHS)
                    print(f"GPU memory summary:\n{torch.cuda.memory_summary(0)}")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20, max_name_column_width=20))
            
    else:
        train_model(models, criterion, optimizers, x_train_subsets, y_train_subsets, enable_print=PRINTING_ENABLED, epochs=EPOCHS)
        if device == 'cpu':
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # in MB
            print(f"CPU memory usage: {memory_usage} MB")
        else:
            print(f"GPU memory summary:\n{torch.cuda.memory_summary(0)}")

    y_lr_train_pred, y_lr_test_pred = [], []
    y_lr_test_pred_np, y_lr_train_pred_np = [], []
    plt.figure(figsize=(5,5))
    cmap = matplotlib.colormaps.get_cmap("tab10")
    for i in range(NUM_SUBSECTIONS):
        y_lr_train_pred = predict(models[i], x_train_subsets[i])
        y_lr_test_pred = predict(models[i], x_test_tensor)
        y_lr_train_pred_np = y_lr_train_pred.cpu().numpy()
        y_lr_test_pred_np = y_lr_test_pred.cpu().numpy()

        # Performance evaluation
        lr_train_mse = mean_squared_error(y_train_subsets[i].to('cpu'), y_lr_train_pred_np)
        lr_train_r2 = r2_score(y_train_subsets[i].to('cpu'), y_lr_train_pred_np)
        lr_test_mse = mean_squared_error(y_test_tensor.to('cpu'), y_lr_test_pred_np)
        lr_test_r2 = r2_score(y_test_tensor.to('cpu'), y_lr_test_pred_np)

        my_print(f'Train set MSE (subset {i + 1}): {lr_train_mse}',
            f'Train set r2 (subset {i + 1}): {lr_train_r2}',
            f'Test set MSE (subset {i + 1}): {lr_test_mse}',
            f'Test set r2 (subset {i + 1} {lr_test_r2}):', sep='\n'
            )

        # Define tolerance level
        tolerance = 0.2
        # Count correct and incorrect predictions with tolerance
        correct_train = ((y_lr_train_pred - y_train_subsets[i]).abs() <= tolerance).sum().item()
        incorrect_train = len(y_train_subsets[i]) - correct_train
        correct_test = ((y_lr_test_pred - y_test_tensor).abs() <= tolerance).sum().item()
        incorrect_test = len(y_test_tensor) - correct_test

        print(f'Train set - Correct Predictions: {correct_train} | Incorrect Predictions: {incorrect_train} | accuracy: {100 * correct_train / (correct_train + incorrect_train)}%')
        print(f'Test set - Correct Predictions: {correct_test} | Incorrect Predictions: {incorrect_test} | accuracy: {100 * correct_test / (correct_test + incorrect_test)}%')

        # Data Visualization
        if PRINTING_ENABLED:
            plt.figure(figsize=(6,6))
            y_test_array = y_test_tensor.squeeze(dim=1).to('cpu').numpy()
            plt.scatter(x=y_test_array, y=y_lr_test_pred_np, alpha=0.3)

            z = np.polyfit(y_test_array, y_lr_test_pred_np, 1)
            # p = np.poly1d(z)

            # plt.plot(y_test_array, p(y_test_array), '#7CAE00')
    if PRINTING_ENABLED:
        plt.ylabel('Predicted Solubility')
        plt.xlabel('Experimental Solubility')
        plt.show()
    
if __name__ == "__main__":
    main()