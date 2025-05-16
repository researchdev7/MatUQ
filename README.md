# MatUQ: Benchmarking GNNs for OOD Materials Property Prediction with Uncertainty Quantification

## Overview
MatUQ is a comprehensive benchmark framework designed for evaluating Graph Neural Networks (GNNs) on out-of-distribution (OOD) materials property prediction tasks with uncertainty quantification (UQ). This repository provides implementations and evaluation scripts described in the MatUQ paper.

## Repository Structure
```
MatUQ/
├── models/                     # Directory containing model implementations and configurations
├── utils.py                    # Utility functions for data processing and evaluation
├── main.py                     # Main script to run experiments
├── main_tf.py                  # Main script for tensor-based implementations
├── main_soap_clusters.py       # Script for SOAP-LOCO splitting strategy
├── main_soap_clusters_tf.py    # Tensor-based SOAP-LOCO implementation
├── config.yml                  # Configuration file for experiment settings
├── environment.yml             # Conda environment file for dependency management
...
```

## Installation

### Clone the Repository
```bash
git clone https://github.com/Sysuzqs/MatUQ.git
cd MatUQ
```

### Set Up the Environment
We recommend using Conda to manage dependencies and environments:
```bash
conda env create -f environment.yml
conda activate matuq
```

## Data Processing and Preparation
MatUQ leverages datasets from six materials databases. To prepare datasets:
1. Download the datasets as specified in the paper.
2. Preprocess the datasets using provided utilities:
```bash
```

## Running Experiments

### Basic Usage
To run experiments with default settings:
```bash
python main.py
```

### Using SOAP-LOCO Splitting
To perform experiments using the SOAP-LOCO splitting strategy:
```bash
python main_soap_clusters.py
```

### Tensor-based Implementation
For tensor-based implementations:
```bash
python main_tf.py
# or for SOAP-LOCO tensor-based implementation
python main_soap_clusters_tf.py
```

## Configuration
Adjust experiment settings through the `config.yml` file:
```yaml
# Example configuration
model:
  name: SchNet
  dropout_rate: 0.1
training:
  epochs: 100
  batch_size: 32
```

## Evaluating Results
The framework automatically evaluates models using standard metrics and the newly proposed D-EviU uncertainty metric. Results are outputted clearly in logs and can be further analyzed using provided scripts.

## Citation
If you use MatUQ in your research, please cite our paper:
```bibtex
@article{MatUQ,
  title={MatUQ: Benchmarking GNNs for OOD Materials Property Prediction with Uncertainty Quantification},
  author={Anonymous Authors},
  year={2025},
}
```
