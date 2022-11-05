import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.nn as nn
from datetime import datetime
import os
import logging
import pickle
# My Modules
from library.utils import makedir, setup_logger, plot_xy, create_figure, SaveBestModel, Progress_images
from library.models import Linear_AE, CNN_AE
from library.train import train, validation

def main():
    log_file_name = 'info.log'
    setup_logger(log_file_name=log_file_name)

    # get date and time
    date = datetime.today().strftime('%Y%m%d_%H%M%S') 

    # create dateset dir
    dir_datasets = makedir('datasets')

    # setup directories for this run
    dir_results = makedir(directory_name='results')
    dir_results_training = makedir(directory_name='training', path_parentdirectory=dir_results)
    dir_run = makedir(directory_name=f'run_{date}', path_parentdirectory=dir_results_training)
    dir_img = makedir(directory_name='images', path_parentdirectory=dir_run)

    # setup transform objects and load datasets
    transform = ToTensor()
    trainset = MNIST(root=dir_datasets, train=True, download=True, transform=transform)
    testset = MNIST(root=dir_datasets, train=False, transform=transform)

    # indices = torch.arange(1000)
    # trainset = Subset(trainset, indices)
    # testset = Subset(testset, indices)

    batchsize = 64
    # create Dataloaders
    trainloader = DataLoader(dataset=trainset, batch_size=batchsize)
    testloader = DataLoader(dataset=testset, batch_size=batchsize)

    # get input size
    input_size = trainloader.dataset[0][0].size()

    model = CNN_AE()
    logging.info('')
    logging.info('---Model Architecure---')
    logging.info(model)
    criterion = nn.MSELoss()
    lr =1e-3
    optimizer = torch.optim.Adam(params=model.parameters() ,lr=lr)
    save_best_model = SaveBestModel()
    num_epochs = 50

    logging.info('')
    logging.info(f'---Dataset Information---')
    logging.info(f'Datasetsize: {len(trainset)+len(testset)}')
    logging.info(f'Trainset size: {len(trainset)}')
    logging.info(f'Testset size: {len(testset)}')
    logging.info('')
    logging.info(f'---Trainingsparameter---')
    logging.info(f'Model: {model.__class__.__name__}')
    # logging.info(f'Encoding size: {embedding_size}')
    logging.info(f'Number of Epochs: {num_epochs}')
    logging.info(f'Train Criterion: {criterion.__class__.__name__}')
    logging.info(f'Optimizer: {optimizer.__class__.__name__}')
    logging.info(f'Learninrate: {lr}')
    logging.info('')

    checkpoint_path = os.path.join(dir_run, 'checkpoint.pt')
    model_path = os.path.join(dir_run, 'model.pt')

    # setup figures for progress results
    progress_image = Progress_images(num_epochs)

    # Training
    dict_losses = {  'epochs': [],
                    'train_losses': [],
                    'validation_losses': []}
    for epoch in range(num_epochs):
        train(model=model, trainloader=trainloader, criterion=criterion, optimizer=optimizer)

        train_loss = validation(model=model, loader=trainloader, criterion=criterion)
        val_loss = validation(model=model, loader=testloader, criterion=criterion)

        save_best_model(val_loss, epoch, model, optimizer, criterion, checkpoint_path)

        logging.info(f'Epoch: {epoch+1} \t train_loss: {train_loss:.6f} \t val_loss: {val_loss:.6f}')

        dict_losses['epochs'].append(epoch+1) 
        dict_losses['train_losses'].append(train_loss) 
        dict_losses['validation_losses'].append(val_loss) 

        # create progress results
        progress_image(model, testset, epoch, flatten=False)

    # Best Validation Loss
    min_val_loss = min(dict_losses['validation_losses'])
    index = dict_losses['validation_losses'].index(min_val_loss)
    best_epoch = dict_losses['epochs'][index]
    logging.info('')
    logging.info(f'Minium Validation loss: {min_val_loss}') 
    logging.info(f'Best Epoch: {best_epoch}') 

    # save best model as TorchScript Format
    model = CNN_AE()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(model_path) # Save

    progress_image.save(dir_img)

    # plot loss epoch charts
    fig = create_figure(xlabel='Epochs',
                        ylabel='Losses',
                        )
    plot_xy(x=dict_losses['epochs'],
            y=dict_losses['train_losses'],
            label='train_losses',
            fig=fig)
    plot_xy(x=dict_losses['epochs'],
            y=dict_losses['validation_losses'],
            label='validation_losses',
            fig=fig,)
    plt.savefig(os.path.join(dir_img, 'loss.svg'))

    # save dataseries of loss epoch cahrts
    with open(os.path.join(dir_run, 'losses.pkl'), 'wb') as f:
        pickle.dump(dict_losses, f)


    # rename dir_run for more information
    dir_run_new = os.path.join(dir_run, os.pardir, f'{model.__class__.__name__}_{date}')
    os.rename(src=dir_run, dst=dir_run_new)
    dir_run = os.path.relpath(dir_run_new)

    # save log file in run dir
    logging.shutdown()
    os.rename(src= log_file_name, dst= os.path.join(dir_run, log_file_name))

if __name__ == "__main__":
    main()
