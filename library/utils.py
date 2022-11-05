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


class Progress_images:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        num_progress_steps = 5
        self.step_size = int(num_epochs/num_progress_steps)
        fig = plt.figure()
        subfigs = fig.subfigures(2,1)
        subfigs[0].suptitle('Ground Truth')
        subfigs[1].suptitle('Reconstruction')
        self.axs_top = subfigs[0].subplots(1)
        if num_epochs%self.step_size == 0:
            self.axs_bottom = subfigs[1].subplots(1, num_progress_steps)
        else:
            self.axs_bottom = subfigs[1].subplots(1, num_progress_steps+1)
        self.cnt = 0

    def __call__(self, model, testset, epoch):
        if epoch%self.step_size == 0 or epoch == self.num_epochs:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            img = testset[0][0].to(device)
            img = torch.flatten(img, start_dim=1)
            recon = model(img)
            recon = torch.reshape(recon,(28, 28))
            recon = recon.cpu().detach().numpy()
            self.axs_bottom[self.cnt].imshow(recon)
            self.axs_bottom[self.cnt].set_title(f'Epoch {epoch+1}')
            self.axs_bottom[self.cnt].axis('off')
            self.cnt += 1
        if epoch == 0:
            # finalize progress results and save it
            img = torch.reshape(img,(28, 28))
            img = img.cpu().detach().numpy()
            self.axs_top.imshow(img)
            self.axs_top.axis('off')

    def save(self, path):
        plt.savefig(os.path.join(path, 'Reconstruction progress.svg'))