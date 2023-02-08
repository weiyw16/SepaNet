# SepaNet: A deep learning-based P- and S-wave separation method for vertical seismic data

[![DOI](https://zenodo.org/badge/324677777.svg)](https://zenodo.org/badge/latestdoi/324677777)

## Background
We propose a data-driven deep learning-based P- and S-wave separation method. Our method employs a neural network that simultaneously extracts P- and S-potential data from multi-components. To avoid the enormous computational cost in wave simulation while constructing training datasets with sufficient kinematic and dynamic variations, we employ a smart wavefield sampling strategy where only a dozen elastic wave simulations are performed on a single velocity model. Generalization tests on various synthetic models and their corresponding reverse time migration images demonstrate that the proposed strategy provides sufficient sampling of the high dimensional data space and virtually ensures successful applications of the trained NN on a vast range of geological scenarios.

This project contains 
1. a runable achitecture of the neural network used for P/S separation
2. a network model that trained on our synthetic data
3. the testing data for a quick run of our method

## Install

Our method bases on [Python3](https://www.python.org/downloads/) and [Pytorch](https://pytorch.org). <br>
First, install your python environment. We recommend [Anaconda](https://www.anaconda.com/).<br>
Second, go into the root directory of this repository, and install the required package in your python3 environment, 

```sh
$ pip install -r requirements.txt
```

## Usage

### train 
first modify main.py and then 

```sh
$ bash ./run.sh main 
```

### test
Modify test.py and then 

```sh
$ bash ./run.sh test 
```
### demo

```sh
$ bash ./run.sh demo
```

## Relative publications

[Deep Learning-Based P- and S-Wave Separation for Multicomponent Vertical Seismic Profiling](https://doi.org/10.1109/TGRS.2021.3124413)<br>
[Yanwen Wei](https://scholar.google.com/citations?hl=en&user=il-IuekAAAAJ&view_op=list_works&sortby=pubdate), Yunyue Elita Li, Jingjing Zong, Jizhong Yang, Haohuan Fu, and Mengyao Sun. 
IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-16, 2022, Art no. 5908116, doi: 10.1109/TGRS.2021.3124413.

[Multi-task learning based P/S wave separation and reverse time migration for VSP](https://doi.org/10.1190/segam2020-3426539.1)<br>
SEG Technical Program Expanded Abstracts 2020. September 2020, 1671-1675

