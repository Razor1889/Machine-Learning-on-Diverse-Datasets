import torch, sys
import matplotlib.pyplot as plt
# master module for all the tests -- used to change units, printing, etc.

PRINTING_ENABLED : bool = False
# internal variable, no need to import
GRAPH_VISIBILITY_ENABLED : bool = True
UNIT_OF_CHOICE : str = 'microseconds'
CONVERSION_FACTOR : int = int(1e6)
SEED_VAL : int = 100
TRAIN_PERCENT : int = 80

LEARN_RATE_CAT : float = 0.001
EPOCHS_CAT : int = 3000

LEARN_RATE : float = 0.01
EPOCHS : int = 1500

SAVE_DIR : str = "models"
RANDOM_STATE : int = 100

NUM_SUBSECTIONS : int = 4

def my_print(*args, sep=' ', end='\n', file=None, flush=False):
    if PRINTING_ENABLED:
        print(*args, sep=sep, end=end, file=file, flush=flush)
    return

def convert_and_round(time_seconds : float) -> float:
    return round(time_seconds * CONVERSION_FACTOR, 4)

# Note that testing takes minimum execution time
def average_time(time_list: list[float]) -> float :
    sums : int = sum([time for time in time_list]) * CONVERSION_FACTOR
    return sums / len(time_list)

def seed() -> None:
    """seeds torch and torch.cuda"""
    torch.cuda.manual_seed(SEED_VAL)
    torch.manual_seed(SEED_VAL)

def plot_predictions(train_features : torch.tensor, train_labels : torch.tensor, test_features : torch.tensor, test_labels : torch.tensor,
                     predictions : torch.tensor = None, alpha : float = 0.3, show_plot=False,
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
    
    # print(train_features_numpy.size, train_labels_numpy.size)
    # return
    
    # plot training data in blue
    plt.scatter(train_features_numpy, train_labels_numpy, c='b', s=4, label="training data", alpha=alpha)
    
    plt.scatter(test_features_numpy, test_labels_numpy, c='r', s=4, label="testing data", alpha=alpha)
    
    if predictions is not None:
        plt.scatter(test_features_numpy, predictions_numpy, c='black', s=4, label="predictions", alpha=alpha)
    plt.legend(prop={"size" : 14})
    plt.title(title)
    if show_plot:
        plt.show()

def check_show_table():
    if len(sys.argv) > 1:
        if any(i in sys.argv for i in ('-s', '--show')):
            return True
    return False
            
def check_print():
    if len(sys.argv) > 1:
        if any(i in sys.argv for i in ('-p', '--print')):
             return True
    return False
