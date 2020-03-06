import argparse
import json

import cloudpickle as cpkl
import networkx as nx
import numpy as np
import tensorflow as tf

from cv import gi_train_test_split
from i_o import get_logger, log_dict, setup_logging
from matrix_completion import KPMF, PMF, KPMF_b, MCScaler, PMF_b

from ngmc import NGMC
from utils import (check_gi_obj, evaluate_model, get_laplacian, get_ppi_data,
                   log_results, sparsity, summarize_results)
from xsmf import (KXSMF, XSMF, KXSMF_b, normalize_sim_scores)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_fraction', type=float, required=True,
                        default=0.01)
    parser.add_argument('--random_seed', type=int, required=False,
                        default=47951)
    parser.add_argument('--n_repeats', type=int, required=True)

    parser.add_argument('--results_output', type=str, required=True)
    parser.add_argument('--models_output', type=str, required=True)
    parser.add_argument('--logfile', type=str, required=True)

    # Subparsers
    subparsers = parser.add_subparsers(dest='mc_alg')
    subparsers.required = True

    add_pmf_arguments(subparsers.add_parser('PMF'))
    add_pmf_b_arguments(subparsers.add_parser('PMF_b'))
    add_kpmf_arguments(subparsers.add_parser('KPMF'))
    add_kpmf_b_arguments(subparsers.add_parser('KPMF_b'))
    add_ngmc_arguments(subparsers.add_parser('NGMC'))

    add_xsmf_arguments(subparsers.add_parser('XSMF'))

    add_kxsmf_arguments(subparsers.add_parser('KXSMF'))
    add_kxsmf_b_arguments(subparsers.add_parser('KXSMF_b'))

    return parser.parse_args()

def add_mf_arguments(parser):
    parser.add_argument('--target_gis', type=str, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--iters', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--report_every', type=int, required=True)

def add_pmf_arguments(parser):
    add_mf_arguments(parser)
    parser.add_argument('--lambda_f', type=float, required=True)

def add_pmf_b_arguments(parser):
    add_mf_arguments(parser)
    parser.add_argument('--lambda_f', type=float, required=True)
    parser.add_argument('--lambda_b', type=float, required=True)


def add_kpmf_arguments(parser):
    add_mf_arguments(parser)
    parser.add_argument('--lambda_f', type=float, required=True)
    parser.add_argument('--lambda_h', type=float, required=True)
    parser.add_argument('--target_ppi', type=str, required=True)
    parser.add_argument('--rl_lambda', type=float, required=True)

def add_kpmf_b_arguments(parser):
    add_mf_arguments(parser)
    parser.add_argument('--lambda_b', type=float, required=True)
    parser.add_argument('--lambda_f', type=float, required=True)
    parser.add_argument('--lambda_h', type=float, required=True)
    parser.add_argument('--target_ppi', type=str, required=True)
    parser.add_argument('--rl_lambda', type=float, required=True)

def add_ngmc_arguments(parser):
    add_mf_arguments(parser)
    parser.add_argument('--lambda_f', type=float, required=True)
    parser.add_argument('--lambda_h', type=float, required=True)
    parser.add_argument('--lambda_p', type=float, required=True)
    parser.add_argument('--alpha_p',  type=float, required=True)
    parser.add_argument('--target_ppi', type=str, required=True)

def add_xsmf_arguments(parser):
    add_mf_arguments(parser)
    parser.add_argument('--source_gis', type=str, required=True)
    parser.add_argument('--sim_scores', type=str, required=True)
    
    parser.add_argument('--lambda_sim', type=float, required=True)
    parser.add_argument('--lambda_src', type=float, required=True)
    parser.add_argument('--lambda_u', type=float, required=True)
    parser.add_argument('--lambda_v', type=float, required=True)
    parser.add_argument('--lambda_us', type=float, required=True)
    parser.add_argument('--lambda_vs', type=float, required=True)

def add_kxsmf_arguments(parser):
    add_xsmf_arguments(parser)
    parser.add_argument('--target_ppi', type=str, required=True)
    parser.add_argument('--source_ppi', type=str, required=True)
    parser.add_argument('--lambda_tgt_rl', type=float, required=True)
    parser.add_argument('--lambda_src_rl', type=float, required=True)


def add_kxsmf_b_arguments(parser):
    add_kxsmf_arguments(parser)
    parser.add_argument('--lambda_b', type=float, required=True)

def train_pmf_model(X_train, rank, iters, lr, lam, report_every):
    pmf = PMF(rank=rank,
              lambda_f = lam,
              use_sigmoid= False,
              max_iter = iters,
              lr = lr,
              report_every = report_every)
    pmf.fit(X_train)
    return pmf

def train_pmf_models(train_Xs, rank, iters, lr, lam, report_every):
    models = [train_pmf_model(X, rank, iters, lr, lam, report_every)
              for X in train_Xs]
    fitted_Xs = [model.X_fitted for model in models]
    factors = [(model.F, model.H) for model in models]
    return fitted_Xs, factors


def train_pmf_b_model(X_train, rank, iters, lr, lam, lam_b, report_every):
    pmf_b = PMF_b(rank=rank,
              lambda_f = lam,
              lambda_b = lam_b,
              use_sigmoid= False,
              max_iter = iters,
              lr = lr,
              report_every = report_every)
    pmf_b.fit(X_train)
    return pmf_b

def train_pmf_b_models(train_Xs, rank, iters, lr, lam, lam_b, report_every):
    models = [train_pmf_b_model(X, rank, iters, lr, lam, lam_b, report_every)
              for X in train_Xs]
    fitted_Xs = [model.X_fitted for model in models]
    factors = [(model.F, model.H) for model in models]
    return fitted_Xs, factors

def train_kpmf_model(X_train, RL, 
                      rank, iters , lr, 
                      lambda_f, lambda_h,
                      report_every):
    kpmf = KPMF(rank = rank,
                lambda_f = lambda_f,
                lambda_h = lambda_h,
                max_iter = iters,
                lr = lr,
                use_sigmoid = False,
                report_every = report_every)
    kpmf.fit(X_train, F_kernel=RL)
    return kpmf

def train_kpmf_models(train_Xs, L, 
                      rank, iters , lr, 
                      lambda_f, lambda_h, rl_lambda,
                      report_every):
    RL = np.linalg.inv(rl_lambda * L + np.eye(len(L)))
    models = [train_kpmf_model(X, RL, 
                                rank, iters , lr, 
                                lambda_f, lambda_h,
                                report_every)
              for X in train_Xs]
    fitted_Xs = [model.X_fitted for model in models]
    factors = [(model.F, model.H) for model in models]
    return fitted_Xs, factors


def train_kpmf_b_model(X_train, RL, 
                      rank, iters , lr, lambda_b,
                      lambda_f, lambda_h,
                      report_every):
    kpmfb = KPMF_b(rank = rank,
                lambda_b = lambda_b,
                lambda_f = lambda_f,
                lambda_h = lambda_h,
                max_iter = iters,
                lr = lr,
                use_sigmoid = False,
                report_every = report_every)
    kpmfb.fit(X_train, F_kernel=RL)
    return kpmfb

def train_kpmf_b_models(train_Xs, L, 
                      rank, iters , lr, lambda_b,
                      lambda_f, lambda_h, rl_lambda,
                      report_every):
    RL = np.linalg.inv(rl_lambda * L + np.eye(len(L)))
    models = [train_kpmf_b_model(X, RL, 
                                rank, iters , lr,
                                lambda_b,
                                lambda_f, lambda_h,
                                report_every)
              for X in train_Xs]
    fitted_Xs = [model.X_fitted for model in models]
    factors = [(model.F, model.H) for model in models]
    return fitted_Xs, factors

def train_ngmc_model(X_train, A,
                      rank, iters, lr, alpha_p,
                      lambda_f, lambda_h, lambda_p):
    ngmc = NGMC(rank=rank, max_iter=iters, alpha=lr, alpha_p=alpha_p,
                lambda_f=lambda_f, lambda_h=lambda_h, lambda_p=lambda_p)
    ngmc.fit(X_train, P=A)
    return ngmc

def train_ngmc_models(train_Xs, A,
                      rank, iters, lr, alpha_p,
                      lambda_f, lambda_h, lambda_p):
    models = [train_ngmc_model(X, A,
                      rank, iters, lr, alpha_p, 
                      lambda_f, lambda_h, lambda_p) for X in train_Xs]
    fitted_Xs = [model.X_fitted for model in models]
    # factors = [(model.F, model.H) for model in models]
    return fitted_Xs, None

def train_xsmf_model(X, X_src, sim_scores,
                     rank, iters, lr,
                     lambda_sim,
                     lambda_src,
                     lambda_u,
                     lambda_v,
                     lambda_us,
                     lambda_vs,
                     report_every):
    xsmf = XSMF(X_tgt=X, X_src=X_src, sim_scores=sim_scores,
                rank=rank, max_iter=iters, lr=lr, 
                lambda_sim=lambda_sim, lambda_src=lambda_src,
                lambda_u=lambda_u, lambda_v=lambda_v,
                lambda_us=lambda_us, lambda_vs=lambda_vs,
                report_every=report_every)
    xsmf = xsmf.fit()
    tf.reset_default_graph()
    return xsmf

def train_xsmf_models(train_Xs, X_src,
                     sim_scores,
                     rank, iters, lr,
                     lambda_sim,
                     lambda_src,
                     lambda_u,
                     lambda_v,
                     lambda_us,
                     lambda_vs,
                     report_every):
    models = [train_xsmf_model(X, X_src, sim_scores,
                                rank, iters , lr, 
                                lambda_sim, lambda_src,
                                lambda_u, lambda_v,
                                lambda_us, lambda_vs,
                                report_every) for X in train_Xs]
    fitted_Xs = [model.X_fitted for model in models]
    factors = [(model.U, model.V, model.US, model.VS) for model in models]
    return fitted_Xs, factors

def train_kxsmf_model(X, X_src, 
                     L_tgt, L_src,
                     sim_scores,
                     rank, iters, lr,
                     lambda_sim,
                     lambda_src,
                     lambda_u,
                     lambda_v,
                     lambda_us,
                     lambda_vs,
                     lambda_tgt_rl,
                     lambda_src_rl,
                     report_every):
    kxsmf = KXSMF(X_tgt=X, X_src=X_src,
                    L_tgt=L_tgt, L_src=L_src,
                    sim_scores=sim_scores,
                    rank=rank, max_iter=iters, lr=lr, 
                    lambda_sim=lambda_sim, lambda_src=lambda_src,
                    lambda_u=lambda_u, lambda_v=lambda_v,
                    lambda_us=lambda_us, lambda_vs=lambda_vs,
                    lambda_tgt_rl=lambda_tgt_rl,
                    lambda_src_rl=lambda_src_rl,
                    report_every=report_every)
    kxsmf = kxsmf.fit()
    tf.reset_default_graph()
    return kxsmf

def train_kxsmf_models(train_Xs, X_src, 
                     L_tgt, L_src,
                     sim_scores,
                     rank, iters, lr,
                     lambda_sim,
                     lambda_src,
                     lambda_u,
                     lambda_v,
                     lambda_us,
                     lambda_vs,
                     lambda_tgt_rl,
                     lambda_src_rl,
                     report_every):
    models = [train_kxsmf_model(X, X_src,L_tgt, L_src,
                                sim_scores,
                                rank, iters, lr,
                                lambda_sim,
                                lambda_src,
                                lambda_u,
                                lambda_v,
                                lambda_us,
                                lambda_vs,
                                lambda_tgt_rl,
                                lambda_src_rl,
                                report_every) for X in train_Xs]
    fitted_Xs = [model.X_fitted for model in models]
    factors = [(model.U, model.V, model.US, model.VS) for model in models]
    return fitted_Xs, factors


def train_kxsmfb_model(X, X_src, 
                     L_tgt, L_src,
                     sim_scores,
                     rank, iters, lr,
                     lambda_b,
                     lambda_sim,
                     lambda_src,
                     lambda_u,
                     lambda_v,
                     lambda_us,
                     lambda_vs,
                     lambda_tgt_rl,
                     lambda_src_rl,
                     report_every):
    kxsmfb = KXSMF_b(X_tgt=X, X_src=X_src,
                    L_tgt=L_tgt, L_src=L_src,
                    sim_scores=sim_scores,
                    rank=rank, max_iter=iters, lr=lr, 
                    lambda_b=lambda_b,
                    lambda_sim=lambda_sim, lambda_src=lambda_src,
                    lambda_u=lambda_u, lambda_v=lambda_v,
                    lambda_us=lambda_us, lambda_vs=lambda_vs,
                    lambda_tgt_rl=lambda_tgt_rl,
                    lambda_src_rl=lambda_src_rl,
                    report_every=report_every)
    kxsmfb = kxsmfb.fit()
    tf.reset_default_graph()
    return kxsmfb

def train_kxsmfb_models(train_Xs, X_src, 
                     L_tgt, L_src,
                     sim_scores,
                     rank, iters, lr,
                     lambda_b,
                     lambda_sim,
                     lambda_src,
                     lambda_u,
                     lambda_v,
                     lambda_us,
                     lambda_vs,
                     lambda_tgt_rl,
                     lambda_src_rl,
                     report_every):
    models = [train_kxsmfb_model(X, X_src,L_tgt, L_src,
                                sim_scores,
                                rank, iters, lr,
                                lambda_b,
                                lambda_sim,
                                lambda_src,
                                lambda_u,
                                lambda_v,
                                lambda_us,
                                lambda_vs,
                                lambda_tgt_rl,
                                lambda_src_rl,
                                report_every) for X in train_Xs]
    fitted_Xs = [model.X_fitted for model in models]
    factors = [(model.U, model.V, model.US, model.VS) for model in models]
    return fitted_Xs, factors

def evaluate_a_pred(true_X, pred_X, mask):
    return evaluate_model(true_X[mask], pred_X[mask])

def evaluate_preds(true_Xs, pred_Xs, test_masks):
    return [evaluate_a_pred(true, predicted, mask) for true, predicted, mask in zip(true_Xs, pred_Xs, test_masks)]

def main():
    args = parse_args()
    setup_logging(args.logfile)

    log = get_logger()
    assert( 0 <= args.hidden_fraction <= 1 )
    
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    args = parse_args()
    log.info('*' * 100)
    log.info('[Starting MC experiment]')
    log_dict(log.info, vars(args))

    log.info('[Loading input data]')

    with open(args.target_gis, 'rb') as f:
        gi_data = cpkl.load(f)

    row_genes = gi_data['rows']

    log.info('\t- setting up training and test sets')
    train_test_sets = [gi_train_test_split(gi_data, args.hidden_fraction) for _ in range(args.n_repeats)]
    
    train_Xs, test_Xs , test_masks= zip(*train_test_sets)
    if args.mc_alg == 'NGMC':
        scalers = [MCScaler('0-1') for _ in range(args.n_repeats)]
    else:
        scalers = [MCScaler('std') for _ in range(args.n_repeats)]

    # if args.mc_alg in ['XSMF, KXSMF']:
    #     train_Xs = [scaler.fit_transform(X).T for scaler, X in zip(scalers, train_Xs)] # Take transposes here for XSMF, KXSMF
    # else:
    #     train_Xs = [scaler.fit_transform(X) for scaler, X in zip(scalers, train_Xs)] # Take transposes here for XSMF, KXSMF

    train_Xs = [scaler.fit_transform(X) for scaler, X in zip(scalers, train_Xs)]

    if args.mc_alg == 'PMF':
        imputed_Xs, models_info = train_pmf_models(train_Xs = train_Xs,
                                                   rank = args.rank,
                                                   iters = args.iters,
                                                   lr = args.lr,
                                                   lam = args.lambda_f,
                                                   report_every = args.report_every)
    elif args.mc_alg == 'PMF_b':
        imputed_Xs, models_info = train_pmf_b_models(train_Xs = train_Xs,
                                                   rank = args.rank,
                                                   iters = args.iters,
                                                   lr = args.lr,
                                                   lam = args.lambda_f,
                                                   lam_b = args.lambda_b,
                                                   report_every = args.report_every)
    elif args.mc_alg == 'KPMF':
        L = get_laplacian(list(row_genes), args.target_ppi)
        imputed_Xs, models_info = train_kpmf_models(train_Xs = train_Xs,
                                                    L = L,
                                                    rank = args.rank,
                                                    iters = args.iters,
                                                    lr = args.lr,
                                                    lambda_f = args.lambda_f,
                                                    lambda_h = args.lambda_h,
                                                    rl_lambda = args.rl_lambda,
                                                    report_every = args.report_every)
    elif args.mc_alg == 'KPMF_b':
        L = get_laplacian(list(row_genes), args.target_ppi)
        imputed_Xs, models_info = train_kpmf_b_models(train_Xs = train_Xs,
                                                    L = L,
                                                    rank = args.rank,
                                                    iters = args.iters,
                                                    lr = args.lr,
                                                    lambda_b = args.lambda_b,
                                                    lambda_f = args.lambda_f,
                                                    lambda_h = args.lambda_h,
                                                    rl_lambda = args.rl_lambda,
                                                    report_every = args.report_every)
    elif args.mc_alg == 'NGMC':
        ppi = nx.read_edgelist(args.target_ppi)
        A = get_ppi_data(list(row_genes), ppi, mode='normalized_adjacency')
        imputed_Xs, models_info = train_ngmc_models(train_Xs = train_Xs,
                                                    A = A,
                                                    rank = args.rank,
                                                    iters = args.iters,
                                                    lr = args.lr,
                                                    alpha_p = args.alpha_p,
                                                    lambda_f = args.lambda_f,
                                                    lambda_h = args.lambda_h,
                                                    lambda_p = args.lambda_p)
    elif args.mc_alg == 'XSMF':
        with open(args.source_gis, 'rb') as f:
            src_gi_data = cpkl.load(f)
        X_src = src_gi_data['values']
        X_src = MCScaler(mode='std').fit_transform(X_src)

        log.info('[Loading sim scores]')
        with open(args.sim_scores, 'rb') as f:
            sim_scores_data = cpkl.load(f)
        sim_scores = sim_scores_data['values']
        sim_scores = sim_scores / np.max(sim_scores) # Normalize

        imputed_Xs, models_info = train_xsmf_models(train_Xs = train_Xs,
                                                    X_src = X_src,
                                                    sim_scores=sim_scores,
                                                    rank = args.rank,
                                                    iters = args.iters,
                                                    lr = args.lr,
                                                    lambda_sim = args.lambda_sim,
                                                    lambda_src = args.lambda_src,
                                                    lambda_u = args.lambda_u,
                                                    lambda_v = args.lambda_v,
                                                    lambda_us = args.lambda_us,
                                                    lambda_vs = args.lambda_vs,
                                                    report_every = args.report_every)
    elif args.mc_alg == 'KXSMF':
        with open(args.source_gis, 'rb') as f:
            src_gi_data = cpkl.load(f)
        X_src = src_gi_data['values']
        X_src = MCScaler(mode='std').fit_transform(X_src)

        log.info('[Loading sim scores]')
        with open(args.sim_scores, 'rb') as f:
            sim_scores_data = cpkl.load(f)
        sim_scores = sim_scores_data['values']
        sim_scores = sim_scores / np.max(sim_scores) # Normalize

        L_tgt = get_laplacian(list(gi_data['rows']), args.target_ppi)
        L_src = get_laplacian(list(src_gi_data['rows']), args.source_ppi)
        log.warn('%s, %s' % L_src.shape)
        log.warn('%s, %s' % X_src.shape)

        imputed_Xs, models_info = train_kxsmf_models(train_Xs = train_Xs,
                                                    X_src = X_src,
                                                    L_tgt=L_tgt,
                                                    L_src=L_src,
                                                    sim_scores=sim_scores,
                                                    rank = args.rank,
                                                    iters = args.iters,
                                                    lr = args.lr,
                                                    lambda_sim = args.lambda_sim,
                                                    lambda_src = args.lambda_src,
                                                    lambda_u = args.lambda_u,
                                                    lambda_v = args.lambda_v,
                                                    lambda_us = args.lambda_us,
                                                    lambda_vs = args.lambda_vs,
                                                    lambda_tgt_rl = args.lambda_tgt_rl,
                                                    lambda_src_rl = args.lambda_src_rl,
                                                    report_every = args.report_every)
    elif args.mc_alg == 'KXSMF_b':
        with open(args.source_gis, 'rb') as f:
            src_gi_data = cpkl.load(f)
        X_src = src_gi_data['values']
        X_src = MCScaler(mode='std').fit_transform(X_src)

        log.info('[Loading sim scores]')
        with open(args.sim_scores, 'rb') as f:
            sim_scores_data = cpkl.load(f)
        sim_scores = sim_scores_data['values']
        sim_scores = sim_scores / np.max(sim_scores) # Normalize

        L_tgt = get_laplacian(list(gi_data['rows']), args.target_ppi)
        L_src = get_laplacian(list(src_gi_data['rows']), args.source_ppi)
        log.warn('%s, %s' % L_src.shape)
        log.warn('%s, %s' % X_src.shape)

        imputed_Xs, models_info = train_kxsmfb_models(train_Xs = train_Xs,
                                                    X_src = X_src,
                                                    L_tgt=L_tgt,
                                                    L_src=L_src,
                                                    sim_scores=sim_scores,
                                                    rank = args.rank,
                                                    iters = args.iters,
                                                    lr = args.lr,
                                                    lambda_b= args.lambda_b,
                                                    lambda_sim = args.lambda_sim,
                                                    lambda_src = args.lambda_src,
                                                    lambda_u = args.lambda_u,
                                                    lambda_v = args.lambda_v,
                                                    lambda_us = args.lambda_us,
                                                    lambda_vs = args.lambda_vs,
                                                    lambda_tgt_rl = args.lambda_tgt_rl,
                                                    lambda_src_rl = args.lambda_src_rl,
                                                    report_every = args.report_every)
    else:
        raise NotImplementedError

    if len(gi_data['rows']) == len(gi_data['cols']) and np.all(gi_data['rows'] == gi_data['cols']):
        log.info('* Averaging over pairs because input is symmetric')
        imputed_Xs = [(X + X.T) / 2 for X in imputed_Xs]
    # if args.mc_alg in ['XSMF, KXSMF']:
    #     imputed_Xs = [scaler.inverse_transform(X).T for scaler, X in zip(scalers, imputed_Xs)] # Take transposes here for XSMF, KXSMF
    # else:
    
    imputed_Xs = [scaler.inverse_transform(X) for scaler, X in zip(scalers, imputed_Xs)] # Take transposes here for XSMF, KXSMF

    results = evaluate_preds(test_Xs, imputed_Xs, test_masks)
    results, fold_results = summarize_results(results)
    log_results(results)

    with open(args.results_output, 'w') as f:
        json.dump(dict(summary=results, collected=fold_results, args=vars(args)), f, indent=2)

    with open(args.models_output, 'wb') as f:
        cpkl.dump(models_info, f)

if __name__ == "__main__":
    main()
