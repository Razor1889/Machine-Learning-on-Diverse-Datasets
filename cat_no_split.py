# imports
# NOTE -- made for windows

# run with -s -p for verbose outputs
import torch
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

# local file
from master import my_print, GRAPH_VISIBILITY_ENABLED, LEARN_RATE_CAT, EPOCHS_CAT,\
    PRINTING_ENABLED, TRAIN_PERCENT, plot_predictions, SAVE_DIR, RANDOM_STATE, seed,\
        check_print, check_show_table
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_epochs():
    if len(sys.argv) > 1:
        return int(sys.argv[len(sys.argv) - 1])
    return EPOCHS_CAT
    
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

# accuracy as a pecentage
def accuracy_percentage(true, pred):
    correct = torch.eq(true, pred).sum().item()
    return (correct / len(pred)) * 100

# Linear Regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, X : torch.Tensor) -> torch.Tensor:
        return self.linear(X)

def train_model(model : nn.Module, criterion, optimizer : optim, x_train : torch.Tensor, y_train : torch.Tensor,
                epochs : int = EPOCHS_CAT, print_num : int = 20, enable_print : bool = False):
    start = perf_counter()
    for epoch in range(epochs):
        # training mode
        model.train()
        # Forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % (epochs // print_num) == 0 and enable_print:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
    print(f"time to train: {perf_counter() - start: .4f} (using {next(model.parameters()).device}, EPOCHS -- {epochs}, LEARN RATE -- {LEARN_RATE_CAT})")

# Predictions
def predict(model : nn.Module, test_data : torch.Tensor):
    model.eval()
    with torch.inference_mode():
        outputs = model(test_data)
        _, predicted = torch.max(outputs.data, 1)
    return predicted

def main() -> int:
    # Load data remotely
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/iris.csv')

    device = get_device()

    # Split data into features (x) and target (y)
    y = df['Species']
    x = df.drop('Species', axis=1)

    # Encode categorical labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=TRAIN_PERCENT / 100, random_state=RANDOM_STATE)
    # Convert data to PyTorch tensors
    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32, device=device)
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)
    # print([x.size() for x in (x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor)])

    # Initialize model, loss function, and optimizer
    input_dim = x_train.shape[1]
    output_dim = len(label_encoder.classes_)
    model = LinearRegressionModel(input_dim, output_dim).to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARN_RATE_CAT)

    # Training the model
    seed()
    PRINTING_ENABLED : bool = check_print()
    SHOW_TABLE : bool = check_show_table()
    EPOCHS_CAT = get_epochs()
    if SHOW_TABLE:
        if device == 'cpu':
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
                with record_function("model_inference"):
                    train_model(model, criterion, optimizer, x_train_tensor, y_train_tensor, enable_print=PRINTING_ENABLED, epochs=EPOCHS_CAT)
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # in MB
                    print(f"CPU memory usage: {memory_usage} MB")
                    
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        else:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
                with record_function("model_inference"):
                    with P.cached():
                        train_model(model, criterion, optimizer, x_train_tensor, y_train_tensor, enable_print=PRINTING_ENABLED, epochs=EPOCHS_CAT)
                    print(f"GPU memory summary:\n{torch.cuda.memory_summary(0)}")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20, max_name_column_width=20))
            
    else:
        train_model(model, criterion, optimizer, x_train_tensor, y_train_tensor, enable_print=PRINTING_ENABLED, epochs=EPOCHS_CAT)
        if device == 'cpu':
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # in MB
            print(f"CPU memory usage: {memory_usage} MB")
        else:
            print(f"GPU memory summary:\n{torch.cuda.memory_summary(0)}")

    y_lr_train_pred = predict(model, x_train_tensor)
    y_lr_test_pred = predict(model, x_test_tensor)

    y_lr_train_pred_np = y_lr_train_pred.cpu().numpy()
    y_lr_test_pred_np = y_lr_test_pred.cpu().numpy()

        # Performance evaluation
    lr_train_mse = mean_squared_error(y_train_tensor.to('cpu'), y_lr_train_pred_np)
    lr_train_r2 = r2_score(y_train_tensor.to('cpu'), y_lr_train_pred_np)
    lr_test_mse = mean_squared_error(y_test_tensor.to('cpu'), y_lr_test_pred_np)
    lr_test_r2 = r2_score(y_test_tensor.to('cpu'), y_lr_test_pred_np)

    my_print(' Train set MSE:', lr_train_mse, '\n',
        'Train set r2:', lr_train_r2, '\n',
        'Test set MSE:', lr_test_mse, '\n',
        'Test set r2:', lr_test_r2
        )

    # Count correct and incorrect predictions
    correct_train = (y_lr_train_pred == y_train_tensor).sum().item()
    incorrect_train = len(y_train_tensor) - correct_train
    correct_test = (y_lr_test_pred == y_test_tensor).sum().item()
    incorrect_test = len(y_test_tensor) - correct_test

    print(f'Train set - Correct Predictions: {correct_train} | Incorrect Predictions: {incorrect_train} | accuracy: {accuracy_percentage(true=y_train_tensor, pred=y_lr_train_pred)}%')
    print(f'Test set - Correct Predictions: {correct_test} | Incorrect Predictions: {incorrect_test} | accuracy: {accuracy_percentage(true=y_test_tensor, pred=y_lr_test_pred)}%')

    # Data Visualization
    if PRINTING_ENABLED:
        plt.figure(figsize=(6,6))
        plt.scatter(x=y_test, y=y_lr_test_pred_np ,alpha=0.3)

        z = np.polyfit(y_test, y_lr_test_pred_np, 1)
        p = np.poly1d(z)

        plt.plot(y_train, p(y_train), '#7CAE00')
        plt.ylabel('Predicted Species')
        plt.xlabel('Experimental Species')
        plt.show()
    
if __name__ == "__main__":
    main()