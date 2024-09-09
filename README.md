# AlexNet Pruning with JAX

## Overview

This project implements iterative pruning and retraining for AlexNet using the JAX framework. The goal is to minimize the number of parameters while maintaining accuracy on the ImageNet dataset.

The pruning technique in this project is inspired by the paper:

**"Learning both Weights and Connections for Efficient Neural Networks"**  
S. Han, J. Pool, J. Tran, W. Dally.  
[arXiv:1506.02626](https://arxiv.org/pdf/1506.02626)

## Project Structure

- **src/**: Contains all the code for dataset loading, model definition, pruning, training, and the main script.
- **doc/**: Placeholder for future documentation.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Project

After installing the project, you can use the CLI `prune-and-train` to run the model with pruning and retraining.

### CLI Usage

```bash
prune-and-train [--prune-ratio <ratio>] [--plot] [--show-pruning-stats]
```
