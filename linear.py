import matplotlib.pyplot as plt
import torch,sys
import numpy as np
from torch import nn
import pathlib
from pathlib import Path
from sklearn.model_selection import train_test_split
from time import perf_counter
from torch.optim.lr_scheduler import ReduceLROnPlateau

WEIGHT : float = 0.7135
BIAS : float = 0.399
EPOCHS : int = int(1e3)

LEARN_RATE : float = 0.1 # determines how big the steps are during training
SAVE_DIR : str = "models"

def get_device() -> str:
    if len(sys.argv) > 1 and 'cuda' in sys.argv and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

device = get_device()

class linearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1) # in and out features are size 1
        
    # forward method needed
    def forward(self, X : torch.Tensor) -> torch.Tensor:
        return self.linear_layer(X)

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

def plot_predictions(train_features : torch.tensor, train_labels : torch.tensor, test_features : torch.tensor, test_labels : torch.tensor,
                     predictions : torch.tensor = None, alpha : float = 0.5, show_plot=False,
                     title : str = "Training vs testing data") -> None:
    """Function to plot test vs train data and predictions using matplotlib.pyplot

    Args:
        train_features (torch.tensor): training features
        train_labels (torch.tensor): training labels (outputs)
        test_features (torch.tensor): test features
        test_labels (torch.tensor): test labels (outputs)
        predictions (torch.tensor, optional): predictions made by your model. Defaults to None.
        alpha (float, optional): sets opacity of graph. Defaults to 0.5.
        show_plot (bool, optional): enables/disables plotting. Defaults to False.
        title (str, optional): title string of graph. Defaults to "Training vs testing data".
    """
    plt.close() # close any old figures
    plt.figure(figsize=(10,7))
    
    train_features_numpy = train_features.detach().cpu().numpy()
    train_labels_numpy = train_labels.detach().cpu().numpy()
    test_features_numpy = test_features.detach().cpu().numpy()
    test_labels_numpy = test_labels.detach().cpu().numpy()
    predictions_numpy = predictions.detach().cpu().numpy() if predictions is not None else None
    
    # plot training data in blue
    plt.scatter(train_features_numpy, train_labels_numpy, c='b', s=4, label="training data", alpha=alpha)
    
    plt.scatter(test_features_numpy, test_labels_numpy, c='r', s=4, label="testing data", alpha=alpha)
    
    if predictions is not None:
        plt.scatter(test_features_numpy, predictions_numpy, c='black', s=4, label="predictions", alpha=alpha)
    plt.legend(prop={"size" : 14})
    plt.title(title)
    if show_plot:
        plt.show()

def get_dataset_size():
    if (len(sys.argv) > 2 and (x in sys.argv for x in ('cuda', 'cpu')) or len(sys.argv > 1)):
        for i in range(1, len(sys.argv)):
            try:
                x = int(sys.argv[i])
                return x
            except ValueError:
                continue
    return 50

def train_and_test_model(epochs : int = EPOCHS, model : torch.nn.Module = None,
                         loss_function = None, optimizer : torch.optim = None,
                         x_test : torch.Tensor = None, x_train : torch.Tensor = None,
                         y_test : torch.Tensor = None, y_train : torch.Tensor = None,
                         print_num : int = 20
                         ) -> tuple[list[int], list[float], list[float]]:
    if any(not x for x in (model, loss_function, optimizer, *(y is not None for y in (x_test, x_train, y_test, y_train)))):
        print("missing argumets for test and train function")
        torch.sys.exit(-1)
    epoch_count = []
    train_loss_values = []
    test_loss_values = []
    start = perf_counter()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    for epoch in range(epochs):
        # set the model to training mode
        model.train() # sets gradients to be True
        
        # 1. Forward pass
        y_pred = model(x_train)
        
        # 2. calculate the loss
        loss = loss_function(y_pred, y_train)
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        
        # 4. Perform back propogation on the loss wrt the parameters of the model 
        loss.backward()
        
        # 5. Step the optimizer (gradient descent)
        optimizer.step()
        # stepping the optimizer will update the parameters
        val_loss = loss_function(model(x_train), y_train)
        scheduler.step(val_loss)
    
        model.eval() # turns off settings in the model which are not needed for evaluation/testing -- dropout, batch norm
        with torch.inference_mode(): # turns off gradient tracking
            y_preds = model(x_test)
            test_loss = loss_function(y_preds, y_test)
        if epoch % (epochs // print_num) == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss)
            test_loss_values.append(test_loss)
            print(f"Epoch: {epoch:04} | Loss: {loss:04.2f} | Test Loss {test_loss:04.2f}")
    time = perf_counter() - start
    print(f"time to train: {time:.4f} (using {next(model.parameters()).device})")
    with open("speeds.txt", 'a') as file:
        file.write(f"{time:.6f}\n")
    return epoch_count, train_loss_values, test_loss_values

def main() -> None:
    start = 0
    end = 1000
    NUM_DATAPOINTS = get_dataset_size()
    step = end / NUM_DATAPOINTS
    X = torch.arange(start,end,step, device=device).unsqueeze(dim=1)
    y = WEIGHT * X + BIAS # y is a linear function of x
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    
    # train_percent : int = 80
    # train_split : int = int((train_percent / 100) * len(X))
    # x_train, y_train = X[:train_split], y[:train_split]
    # x_test, y_test = X[train_split:], y[train_split:]
    print(tuple(len(x) for x in (x_train, y_train, x_test, y_test)))
    plot_predictions(train_features=x_train, train_labels=y_train,
                     test_features=y_test, test_labels=y_test, title="before training (ideal model)")
    torch.manual_seed(42)
    # create the nn module
    model_0 = linearRegressionModel()
    model_0.cpu() if device == "cpu" else model_0.cuda()

    print(model_0, model_0.state_dict())
    # print("\nmaking initial predictions on random data:\n")
    with torch.inference_mode():
        y_preds = model_0(x_test)
        # print(y_test, y_preds, sep='\n')
    plot_predictions(train_features=x_train, train_labels=y_train,
                     test_features=y_test, test_labels=y_test, predictions=y_preds)
    
    # a loss function is used to check model accuracy
    loss_function = nn.L1Loss() # mean average error loss function
    # optimizer and loss function work together

    # SGD (stochastic gradient descent), Adam are popular optimizers
    optimizer = torch.optim.SGD(params=model_0.parameters(), # the params we would like to optimize
                                lr=LEARN_RATE # learning rate (how much the values are changed by)
                                )
    
    # epoch dictates a loop through the data -- a hyperparameter
    torch.manual_seed(42)
    # track different values
    print(f"\nBeginning model training -- {NUM_DATAPOINTS} datasets\n")
    # use a training loop and a testing loop
    epoch_count, train_loss_values, test_loss_values = train_and_test_model(model=model_0, loss_function=loss_function,
                                                                            optimizer=optimizer,
                                                                            x_test=x_test, x_train=x_train, 
                                                                            y_test=y_test, y_train=y_train)
    
    model_0.eval()
    with torch.inference_mode(): # turns off gradient tracking
        y_preds = model_0(x_test)
    
    plt.close()
    plt.plot(torch.tensor(epoch_count), torch.tensor(train_loss_values), label='Train Loss')
    plt.plot(torch.tensor(epoch_count), torch.tensor(test_loss_values), label='Test Loss')
    plt.title("Training and test loss curves")
    plt.ylabel("Loss values")
    plt.xlabel("Epochs")
    plt.legend()
    # plt.show()
    
    print(*(f"{key:10}" for key in model_0.state_dict().keys()), sep=' | ')
    print(*(f"{float(val):<10.3f}" for val in model_0.state_dict().values()), sep=' | ')
    plot_predictions(train_features=x_train, train_labels=y_train,
                     test_features=x_test, test_labels=y_test, predictions=y_preds, show_plot=False)
    
    # save in case we want to load later
    save_model(model_name="linear_regression_model.pth", model_to_save=model_0)

if __name__ == '__main__':
    # with cProfile.Profile() as profile:
    main()
    # results = pstats.Stats(profile)
    # results.sort_stats(pstats.SortKey.TIME)
    # results.print_stats() # too much output