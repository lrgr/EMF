# Extensible Matrix Factorization (EMF)
[![GitHub license](https://img.shields.io/github/license/lrgr/EMF.svg)](https://github.com/lrgr/emf/blob/master/LICENSE)

![EMF schematic diagram](../../blob/master/emf_fig1.png)

Beta release EMF and other baselines are implemented with TensorFlow v1.

## Setup:

We recommend users to install required packages dependencies for the EMF models and experiments, using [Conda](https://conda.io/miniconda.html). To install Python and other dependencies, which you can do directly using the provided `environment.yml file`:

    conda env create -f environment.yml
    source activate EMF

The `xsmf` Conda environment can be used to run any number of experiments and examples included in this project/repository. We note that experiments and data download for other parts of this project are implemented and configured with [Snakemake](http://snakemake.readthedocs.io/en/stable/) which will be installed as part of the `EMF` environment.

### To run models using GPUs:

To use GPUs, install the `xsmf-gpu` environment via:

    conda env create -f environment-gpu.yml
    source activate EMF-gpu

## Experiments:

### 10 fold Monte-Carlo cross-validation benchmarks of matrix factorization models

See `experiments/all-mf-monte-carlo-cv`.

### Hyperparameter search with `hyperopt`

Hyperparameter searches for different MF models are implemented and can be run using `snakemake` in the following directories in `experiments/`:

- `single-species-mf-hp-search` - MF, KPMF, MF with bias, KPMF with bias, NGMC models.
- `xsmf-hp-search` - XSMF model
- `k-xsmf-hp-search` - Kernelized XSMF model
- `k-xsmf-b-hp-search` - Kernelized XSMF with bias model
