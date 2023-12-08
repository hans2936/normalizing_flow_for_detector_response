# Generative Machine Learning for Detector Response Modeling with a Conditional Normalizing Flow
This repository contains the code for [Generative Machine Learning for Detector Response Modeling with a Conditional Normalizing Flow](https://arxiv.org/abs/2303.10148). The data can be found [here](https://drive.google.com/file/d/1SK8sid69tpJGgMq1yi1vjdTEiKvUl1q5/view?usp=sharing).

## Installation
```bash
conda create --name <env_name> python=3.9 pip
conda activate <env_name>
pip install -e .
```

## Instructions
To train a conditional normalizing flow, use ```train_cond.py``` and specify the configuration file. For generation, use ```generate.py``` and specify the configuration file and model. Optional arguments include the epoch and dataset (test, validation, or train) to use for generation. Use ```plot.ipynb``` to produce plots for generation.
```bash
python train_cond.py --config_file config_nf.yml --log-dir <model_name> --epochs 100
python generate.py --config_file config_nf.yml --log-dir <model_name> --epochs-total 100
```
Note: 
- by default, `<model_name>` is `NF` in the `config_nf.yml`
- number of epochs can be increased to imporve the performance

## Configuration file
The configuration can be found in `config_nf.yml` where the following can be specified:
- file_name: list of paths to data files
- out_branch_names: branches to generate or used to calculate the quantity to generate
- truth_branch_names: branches with conditional inputs
- data_branch_names: branches with additional information

Hyperparameters:
- lr: 0.001
- batch_size: 512
- num_layers: 2
- latent_size: 128
- num_bijectors: 10
- hidden_activation: relu

## Code
- preprocess.py: load and preprocess data
- made.py: model definition and construction
- utils.py: loss functions
- utils_plot.py: plotting functions
- train_cond.py: train model and evaluate on validation set
- generate.py: generate detector response for specific epoch of model on the train, validation, or test set
- plot.ipynb: plot generation
