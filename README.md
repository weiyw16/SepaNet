# SepaNet: 
A machine learning-based P- and S-wave separation method for vertical seismic data

## Background
Vertical Seismic Profiling(VSP) in oil and gas exploration helps to derive high-resolution images of the target reservoir and is a cost-effective technique for $CO_2$ storage monitoring. P/S wave separation for VSP is a significant signal processing step to extract independent single-mode waves. Conventional wave separation methods are insufficient for complex geology due to velocity assumptions and manually set parameters. We propose a data-driven deep learning-based P- and S-wave separation method. Our method employs a neural network that simultaneously extracts P- and S-potential data from multi-components. To avoid the enormous computational cost in wave simulation while constructing training datasets with sufficient kinematic and dynamic variations, we employ a smart wavefield sampling strategy where only a dozen elastic wave simulations are performed on a single velocity model. Generalization tests on various synthetic models and their corresponding reverse time migration images demonstrate that the proposed strategy provides sufficient sampling of the high dimensional data space and virtually ensures successful applications of the trained NN on a vast range of geological scenarios.

This project contains 
1. a runable achitecture of the neural network used for P/S separation
2. a network model that trained on our synthetic data
3. the testing data for a quick run of our method

## Install

Our method bases on [python3](https://www.python.org/downloads/) and [pytorch](https://pytorch.org). 
First install your python environment. We recommend [Anaconda] (https://www.anaconda.com/).
Second, get into the ```/src``` directory, and install the required package in your python3 environment, 

```sh
$ pip install -r requirements.txt
```

## Usage

### train 

```sh
$ bash ./run_train.sh [your training data path]
```

### test
```sh
$ bash ./run_test.sh [your testing data path]
```

## Relative articles





