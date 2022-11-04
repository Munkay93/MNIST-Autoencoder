import torch
import os
import logging
import matplotlib.pyplot as plt

def makedir(directory_name: str, path_parentdirectory: str = './') -> str:
    """creates a directory

    Args:
        directory_name (str): _description_
        path_parentdirectory (str, optional): _description_. Defaults to './'.

    Returns:
        str: _description_
    """
    path = os.path.join(path_parentdirectory, directory_name) 
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)
    
    return path

def setup_logger(log_file_name: str):
    """setups a logger and a log file

    Args:
        log_file_name (str): _description_
    """
    try:
        os.remove(log_file_name)
    except OSError as error:
        print(error)
    logging.basicConfig(filename=log_file_name, level=logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logging.getLogger('').addHandler(ch)

    logging.info('Started')


def plot_xy(x: list,
            y:list,
            label: str,
            fig: object,
            legend: bool=True):

    plt.figure(fig.number)
    plt.plot(x, y, label=label)
    if legend:
        plt.legend()


def create_figure(xlabel: str,
                  ylabel: str,
                  titel: str = None,
                  grid: bool=True,
                  ) -> object:

    fig = plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.suptitle(titel)
    if grid:
        plt.grid()
    return fig

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion, model_path
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, model_path)