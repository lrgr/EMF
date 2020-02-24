# Cross-species Matrix Factorization (XSMF)
[![GitHub license](https://img.shields.io/github/license/lrgr/xsmf.svg)](https://github.com/lrgr/xsmf/blob/master/LICENSE)

Beta release. XSMF and other baselines are inmplemented with TensorFlow v1.

## Setup:

We recommend users to install required packages dependencies for the XSMF models and experiments, using [Conda](https://conda.io/miniconda.html). To install Python and other dependencies, which you can do directly using the provided `environment.yml file`:

    conda env create -f environment.yml
    source activate xsmf

This `xsmf` Conda environment can be used to run any number of experiments and examples included in this project/repository. We note that experiments and data download for other parts of this project are implemented and configured with [Snakemake](http://snakemake.readthedocs.io/en/stable/) which will be installed as part of the `xsmf` environment.

### To run models using GPUs:

To use GPUs, install the `xsmf-gpu` environment via:
    conda env create -f environment-gpu.yml
    source activate xsmf-gpu

### Notes
- KXSMF2 and XMSF2 models correspond to similarity score regularized cross-species models.
