# Hyperparameter search for single-species matrix completion models

The following single-species matrix factorization algorithms were benchmarked for genetic interaction (GI) prediction:

- (Probabilistic) matrix factorization (PMF) [1-2]
- PMF with bias (PMF-b) [2]
- Kernelized probabilistic matrix factorization (KPMF) [3]
- KPMF with bias (KPMF-b)
- Network guided matrix factorization (NGMC) [4]

(All models except NGMC is trained using the ADAM optimizer.)

Matrix factorization methods for matrix completion (for genetic interactions) are sensitive to hyper-parameter selection. Careful hyper-parameter selection, and early-exiting criteria improve generalization error.

Thus, we use **Hyperopt** for automatic hyper-parameter selection [5]. The basic execution of hyper-parameter search is configured using Snakemake, and can be executed using the commands described below.

## 1. Basic execution

For each combination of algorithm and dataset, hyper-parameter search is performed for scenarios where 10%, 25%, 50%, and 75% of GI scores for unique gene pairs are used during training. Hyper-parameters are chosen based on performance of a validation set (a further 10% of training examples is held out for this set). Paramters and hyper-parameter ranges required for execution are specified in configuration files in YAML format in `configs/`.

Run a hyper-parameter search with:

	snakemake all --configfile configs/<alg>_<dataset>.yml

For example,

	snakemake all --configfile configs/kpmfb_collins.yml

Performs hyper-parameter selection for KPMF-b models trained on the Collins et al. (Nature 2008) baker's yeast dataset.

## 2. Priors for sampling hyper-parameters

50 iterations of hyperopt is used for each algorithm, dataset, held out rate, combination. The tables below describe the priors from which hyper-parameters are drawn. The configuration files in YAML format, in `configs/`, configure each run and contain keys that correspond to requisite hyper-parameters for each algorithm.

For all algorithms,  hyperopt, searches over ranks over a specified `rank_range` (with `rank_step` intevals) with uniform prior.

### PMF
| Parameter | Prior  | Description
|--|--|--|
| `lr_range` | Log uniform | Learning rate for ADAM |
| `lambda_range` | Log uniform | Scaling term for L2 regularization of latent factors | 

### PMF-b

Same as PMF plus the following:
| Parameter | Prior  | Description
|--|--|--|
| `lambda_b_range` | Log uniform | Scaling term for L2 regularization of learned bias term | 

### KPMF
| Parameter | Prior  | Description
|--|--|--|
| `lambda_f_range` | Log uniform | Scaling term for L2 regularization of LHS latent factor| 
| `lambda_h_range` | Log uniform | Scaling term for quadratic-trace term regularizer of RHS latent factor |
| `rl_lambda_range` | Log uniform | Hyper-parameter for the Regularized Laplacian kernel | 


### KPMF-b
Same as KPMF plus the following:
| Parameter | Prior  | Description
|--|--|--|
| `lambda_b_range` | Log uniform | Scaling term for L2 regularization of learned bias term |

### NGMC
| Parameter | Prior  | Description
|--|--|--|
| `lambda_f_range` | Log uniform | Scaling term for L2 regularization of LHS latent factor| 
| `lambda_h_range` | Log uniform | Scaling term for quadratic-trace term regularizer of RHS latent factor |
| `lambda_p_range` | Log uniform | Scaling term for regularizer that arises from recursive definition of latent factors | 


## 3. Early exit strategy for training MF models

All models are trained with at most `max_iter` iterations (defaults to 500) as specified by the requisite YAML configuration file in `configs/`. All models except NGMC early exit from training if validation time R^2 score has not increased for `early_exit` (defaults to 5) number of iterations.

## 4. Data for MF models that incorporate PPI networks

For KPMF, KPMF-b, and NGMC models, requisite PPI networks from BioGRID (as specified in `data/`) are used to regularize factorization. For KPMF-b and KPMF, the regularized laplacian kernel on the subgraph restricted to genes in the corresponding axis of the input genetic interaction dataset is used. For NGMC, the normalized adjacency matrix on the subgraph restricted to genes corresponding axis of the input genetic interaction dataset is used.

## References
1. Salakhutidinov and Mnih, NeurIPS 2008
2. Koren et al., Computer 2009
3. Zhou et al., SDM 2012
4. Zitnik and Zuppan, Journal of Comp. Bio. 2015
5. Bergstra et al., NeurIPS 2011
