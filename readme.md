# MNIST-Autoencoder

## Training Linear AE

**---Dataset Information---**
Datasetsize: 70000/
Trainset size: 60000/
Testset size: 10000/

**---Trainingsparameter---**
Model: Linear_AE/
Encoding size: 4/
Number of Epochs: 50/
Train Criterion: MSELoss/
Optimizer: Adam/
Learninrate: 0.001/

![alt text](./results/training/01_Examples/linear_AE_example/images/loss.svg)

![alt text](./results/training/01_Examples/linear_AE_example/images/Reconstruction%20progress.svg)

## Inference Linear AE

![alt text](./results/inference/01_Examples/linear_AE_example/results.svg)

## Training CNN AE
Training parameters:

**---Dataset Information---**
Datasetsize: 70000/
Trainset size: 60000/
Testset size: 10000/

**---Trainingsparameter---**
Model: CNN_AE/
Number of Epochs: 50/
Train Criterion: MSELoss/
Optimizer: Adam/
Learninrate: 0.001/

![alt text](./results/training/01_Examples/CNN_AE_example/images/loss.svg)
![alt text](./results/training/01_Examples/CNN_AE_example/images/Reconstruction%20progress.svg)

## Inference CNN AE

![alt text](./results/inference/01_Examples/CNN_AE_example/results.svg)
## TODO

- [x] save training parameters for each run 
- [ ] create table with training parameters in readme
- [x] reshape readme images 
