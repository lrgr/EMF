import json
import numpy as np
import pandas as pd
import networkx as nx

import tensorflow as tf

from sklearn.externals import joblib

import hyperopt
from i_o import get_logger, log_dict, setup_logging
from matrix_completion import KPMF, KPMF_b, PMF, PMF_b, MCScaler, mc_train_test_split
from ngmc import NGMC
from utils import (evaluate_model, check_gi_obj, get_ppi_data, 
                   summarize_results, log_results)

from cv import gi_train_test_split

###############################################################################
#                   Argument Parser(s)
###############################################################################

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    
    # Main parser
    parser.add_argument('-if', '--input_file', type=str, required=True)
    parser.add_argument('-of', '--results_output', type=str, required=True)
    parser.add_argument('-tc', '--training_curve_output', type=str, required=True)
    parser.add_argument('-hp', '--trials_output', type=str, required=True)
    parser.add_argument('-m', '--models_output', type=str, required=True)
    parser.add_argument('-log', '--logfile', type=str, default='/tmp/log.txt')

    parser.add_argument('--n_repeats', type=int, required=False, default=5)
    parser.add_argument('--n_hyperopt_iters', type=int, required=False, default=50)
    parser.add_argument('-hf', '--hidden_fraction', type=float, required=False,
                        default=0.2)
    parser.add_argument('-vhf', '--val_hidden_fraction', type=float, required=False, default=0.2)
    parser.add_argument('-rs', '--random_seed', type=int, required=False,
                        default=47951)
    parser.add_argument('--logistic', action='store_true')
    # Subparsers
    subparsers = parser.add_subparsers(dest='mc_alg')
    add_pmf_arguments(subparsers.add_parser('PMF'))
    add_pmf_b_arguments(subparsers.add_parser('PMF_b'))
    add_kpmf_arguments(subparsers.add_parser('KPMF'))
    add_kpmfb_arguments(subparsers.add_parser('KPMF_b'))
    add_ngmc_arguments(subparsers.add_parser('NGMC'))

    return parser.parse_args()

def add_mf_arguments(parser):
    parser.add_argument('--rank_range', type=int, nargs=2, default=[10, 70])
    parser.add_argument('--rank_step', type=int, default=5)
    parser.add_argument('--max_iter', type=int, default=300)
    parser.add_argument('--lr_range', type=float, nargs=2, default=[0.01, 0.5])
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--report_every', type=int, default=-1)

def add_pmf_arguments(parser):
    add_mf_arguments(parser)
    parser.add_argument('--lambda_range', type=float, nargs=2, default=[0.03, 0.5])

def add_pmf_b_arguments(parser):
    add_mf_arguments(parser)
    parser.add_argument('--lambda_range', type=float, nargs=2, default=[0.03, 0.5])
    parser.add_argument('--lambda_b_range', type=float, nargs=2, default=[0.03, 0.5])

def add_kpmf_arguments(parser):
    add_mf_arguments(parser)
    parser.add_argument('--lambda_f_range', type=float, nargs=2, default=[0.03, 0.5])
    parser.add_argument('--lambda_h_range', type=float, nargs=2, default=[0.03, 0.5])
    parser.add_argument('--ppi', type=str, required=True)
    parser.add_argument('--rl_lambda_range', type=float, nargs=2, default=[0.01,0.5])

def add_kpmfb_arguments(parser):
    add_mf_arguments(parser)
    parser.add_argument('--lambda_f_range', type=float, nargs=2, default=[0.03, 0.5])
    parser.add_argument('--lambda_h_range', type=float, nargs=2, default=[0.03, 0.5])
    parser.add_argument('--lambda_b_range', type=float, nargs=2, default=[0.03, 0.5])
    parser.add_argument('--ppi', type=str, required=True)
    parser.add_argument('--rl_lambda_range', type=float, nargs=2, default=[0.01,0.5])


def add_ngmc_arguments(parser):
    add_mf_arguments(parser)
    parser.add_argument('--lambda_f_range', type=float, nargs=2, default=[0.005, 0.015])
    parser.add_argument('--lambda_h_range', type=float, nargs=2, default=[0.005, 0.015])
    parser.add_argument('--ppi', type=str, required=False)
    parser.add_argument('--lambda_p_range', type=float, nargs=2, default=[0.005, 0.015])
    parser.add_argument('--alpha_p_range', type=float, nargs=2, default=[0.001, 0.002])

###############################################################################
#                   Functions to run PMF
###############################################################################

def pmf_param_space(args):
    '''
    '''
    param_space = {
        'lambda_f': hyperopt.hp.loguniform('lambda_f', 
                                            np.log(args.lambda_range[0]), 
                                            np.log(args.lambda_range[1])),
        'lr': hyperopt.hp.loguniform('lr', 
                                     np.log(args.lr_range[0]), 
                                     np.log(args.lr_range[1])),
        'rank': hyperopt.hp.quniform('rank', 
                                     args.rank_range[0],
                                     args.rank_range[1],
                                     args.rank_step),
        'max_iter': hyperopt.hp.choice('max_iter', [args.max_iter]),
        'early_stop': hyperopt.hp.choice('early_stop', [args.early_stop]),
        'use_sigmoid': args.logistic
    }
    return param_space

def train_pmf(X_train, params, X_val=None,  fit_params=None):
    '''
    Train a PMF model with given training and validation data
    '''
    assert fit_params is None

    # Need to convert rank to int
    if 'rank' in params:
        params['rank'] = np.int(params['rank'])
    pmf_model = PMF(**params)
    pmf_model.fit(X_train, X_val=X_val)
    return pmf_model

def pmf_objective(X_train, X_val, fit_params=None):
    '''
    Return an objective function that that computes R2 of the trained
    PMF model on validation data given model hyperparameters
    '''
    assert fit_params is None

    def obj(params):
        # Note that the parameter dictionary returned is specified *explicitly* with no
        # optional arguments. This makes reporting and retraining easy and unambiguous
        pmf_model = train_pmf(X_train, params, X_val=X_val)
        r2_loss = pmf_model.score(X_val)
        params = {**params, **pmf_model.get_params()}
        params['max_iter'] = pmf_model.iters_ran
        return {
            'loss': 1 - r2_loss,
            'status': hyperopt.STATUS_OK,
            'params': params,
        }
    return obj

def run_pmf(X, fit_params, param_space,
            val_hidden_fraction,
            hidden_fraction, 
            n_repeats,
            hyperopt_iters = 10,
            seed=None,
            logistic=True):

    assert fit_params is None

    log = get_logger()
    log.info('[Training PMF model]')
    scaler = MCScaler(mode='0-1' if logistic else 'std')
    use_validation=True
    return run_mc_alg(X,
                    #   fold_objective=pmf_objective_logistic if logistic else pmf_objective,
                      fold_objective=pmf_objective,
                      fit_params=fit_params,
                    #   retrain_model = train_pmf_logistic if logistic else train_pmf,
                      retrain_model = train_pmf,
                      space=param_space,
                      scaler=scaler,
                      val_hidden_fraction=val_hidden_fraction,
                      hidden_fraction=hidden_fraction,
                      train_with_validation=use_validation,
                      n_repeats=n_repeats,
                      hyperopt_iters=hyperopt_iters,
                      hyperopt_seed=seed)

###############################################################################
#                   Functions to run PMF-b
###############################################################################

def pmfb_param_space(args):
    '''
    '''
    param_space = {
        'lambda_f': hyperopt.hp.loguniform('lambda_f', 
                                            np.log(args.lambda_range[0]), 
                                            np.log(args.lambda_range[1])),
        'lambda_b': hyperopt.hp.loguniform('lambda_b', 
                                            np.log(args.lambda_b_range[0]), 
                                            np.log(args.lambda_b_range[1])),
        'lr': hyperopt.hp.loguniform('lr', 
                                     np.log(args.lr_range[0]), 
                                     np.log(args.lr_range[1])),
        'rank': hyperopt.hp.quniform('rank', 
                                     args.rank_range[0],
                                     args.rank_range[1],
                                     args.rank_step),
        'max_iter': hyperopt.hp.choice('max_iter', [args.max_iter]),
        'early_stop': hyperopt.hp.choice('early_stop', [args.early_stop]),
        'use_sigmoid': args.logistic
    }
    return param_space

def train_pmfb(X_train, params, X_val=None,  fit_params=None):
    '''
    Train a PMF model with given training and validation data
    '''
    assert fit_params is None

    # Need to convert rank to int
    if 'rank' in params:
        params['rank'] = np.int(params['rank'])
    pmf_model = PMF_b(**params)
    pmf_model.fit(X_train, X_val=X_val)
    return pmf_model

def pmfb_objective(X_train, X_val, fit_params=None):
    '''
    Return an objective function that that computes R2 of the trained
    PMF model on validation data given model hyperparameters
    '''
    assert fit_params is None

    def obj(params):
        # Note that the parameter dictionary returned is specified *explicitly* with no
        # optional arguments. This makes reporting and retraining easy and unambiguous
        pmf_model = train_pmfb(X_train, params, X_val=X_val)
        r2_loss = pmf_model.score(X_val)
        params = {**params, **pmf_model.get_params()}
        params['max_iter'] = pmf_model.iters_ran
        return {
            'loss': 1 - r2_loss,
            'status': hyperopt.STATUS_OK,
            'params': params,
        }
    return obj

def run_pmfb(X, fit_params, param_space,
            val_hidden_fraction,
            hidden_fraction, 
            n_repeats,
            hyperopt_iters = 10,
            seed=None,
            logistic=True):

    assert fit_params is None

    log = get_logger()
    log.info('[Training PMF_b model]')
    scaler = MCScaler(mode='0-1' if logistic else 'std')
    use_validation=True
    return run_mc_alg(X,
                      fold_objective=pmfb_objective,
                      fit_params=fit_params,
                      retrain_model = train_pmfb,
                      space=param_space,
                      scaler=scaler,
                      val_hidden_fraction=val_hidden_fraction,
                      hidden_fraction=hidden_fraction,
                      train_with_validation=use_validation,
                      n_repeats=n_repeats,
                      hyperopt_iters=hyperopt_iters,
                      hyperopt_seed=seed)

###############################################################################
#                   Functions to run KPMF+bias
###############################################################################
def kpmfb_param_space(args):
    '''
    '''
    param_space = {
        'lambda_f': hyperopt.hp.loguniform('lambda_f', 
                                            np.log(args.lambda_f_range[0]), 
                                            np.log(args.lambda_f_range[1])),
        'lambda_h': hyperopt.hp.loguniform('lambda_h', 
                                            np.log(args.lambda_h_range[0]), 
                                            np.log(args.lambda_h_range[1])),
        'lambda_b': hyperopt.hp.loguniform('lambda_b', 
                                            np.log(args.lambda_b_range[0]), 
                                            np.log(args.lambda_b_range[1])),
        'rl_lambda': hyperopt.hp.loguniform('rl_lambda', 
                                            np.log(args.rl_lambda_range[0]), 
                                            np.log(args.rl_lambda_range[1])),
        'lr': hyperopt.hp.loguniform('lr', 
                                     np.log(args.lr_range[0]), 
                                     np.log(args.lr_range[1])),
        'rank': hyperopt.hp.quniform('rank', 
                                     args.rank_range[0],
                                     args.rank_range[1],
                                     args.rank_step),
        'max_iter': hyperopt.hp.choice('max_iter', [args.max_iter]),
        'early_stop': hyperopt.hp.choice('early_stop', [args.early_stop]),
        'use_sigmoid': args.logistic
    }
    return param_space

def train_kpmfb(X_train, params, fit_params, X_val=None, ):
    '''
    Train a PMF model with given training and validation data.
        `fit_params`:  contains dictionary containing side information (kernel)
                       used sat training time.
    '''
    # Copy parameters since we need to remove keys, otherwise, deleting keys will
    # mutate the dictionary used in other functions
    _params = params.copy() 

    # Look at parameters to generate kernel
    L = fit_params['L']
    rl_lambda = _params['rl_lambda']
    RL = np.linalg.inv(np.eye(len(L)) + (rl_lambda * L))
    _params.pop('rl_lambda', None)

    # Convert rank to int
    if 'rank' in _params:
        _params['rank'] = np.int(_params['rank'])
    kpmfb_model = KPMF_b(**_params)
    kpmfb_model.fit(X_train, X_val=X_val, F_kernel=RL)
    return kpmfb_model

def kpmfb_objective(X_train, X_val, fit_params):
    '''
    Return an objective function that that computes R2 of the trained
    PMF model on validation data given model hyperparameters
    '''
    assert fit_params is not None
    def obj(params):
        # Note that the parameter dictionary returned is specified *explicitly* with no
        # optional arguments. This makes reporting and retraining easy and unambiguous
        kpmfb_model = train_kpmfb(X_train, params, fit_params=fit_params, X_val=X_val, )
        r2_loss = kpmfb_model.score(X_val)
        params = {**params, **kpmfb_model.get_params()}
        print(params)
        params['max_iter'] = kpmfb_model.iters_ran
        return {
            'loss': 1 - r2_loss,
            'status': hyperopt.STATUS_OK,
            'params': params,
        }
    return obj

def run_kpmfb(X, fit_params, param_space,
             val_hidden_fraction,
             hidden_fraction, 
             n_repeats,
             hyperopt_iters = 10,
             seed=None,
             logistic=True):
    assert fit_params is not None
    log = get_logger()
    log.info('[Training KPMF model]')

    scaler = MCScaler(mode='0-1' if logistic else 'std')
    
    use_validation=True

    return run_mc_alg(X,
                      fold_objective= kpmfb_objective,
                      retrain_model = train_kpmfb,
                      fit_params = fit_params,
                      space=param_space,
                      scaler=scaler,
                      val_hidden_fraction=val_hidden_fraction,
                      hidden_fraction=hidden_fraction,
                      train_with_validation=use_validation,
                      n_repeats=n_repeats,
                      hyperopt_iters=hyperopt_iters,
                      hyperopt_seed=seed)

###############################################################################
#                   Functions to run KPMF
###############################################################################
def kpmf_param_space(args):
    '''
    '''
    param_space = {
        'lambda_f': hyperopt.hp.loguniform('lambda_f', 
                                            np.log(args.lambda_f_range[0]), 
                                            np.log(args.lambda_f_range[1])),
        'lambda_h': hyperopt.hp.loguniform('lambda_h', 
                                            np.log(args.lambda_h_range[0]), 
                                            np.log(args.lambda_h_range[1])),
        'rl_lambda': hyperopt.hp.loguniform('rl_lambda', 
                                            np.log(args.rl_lambda_range[0]), 
                                            np.log(args.rl_lambda_range[1])),
        'lr': hyperopt.hp.loguniform('lr', 
                                     np.log(args.lr_range[0]), 
                                     np.log(args.lr_range[1])),
        'rank': hyperopt.hp.quniform('rank', 
                                     args.rank_range[0],
                                     args.rank_range[1],
                                     args.rank_step),
        'max_iter': hyperopt.hp.choice('max_iter', [args.max_iter]),
        'early_stop': hyperopt.hp.choice('early_stop', [args.early_stop]),
        'use_sigmoid': args.logistic
    }
    return param_space

def train_kpmf(X_train, params, fit_params, X_val=None, ):
    '''
    Train a PMF model with given training and validation data.
        `fit_params`:  contains dictionary containing side information (kernel)
                       used sat training time.
    '''
    # Copy parameters since we need to remove keys, otherwise, deleting keys will
    # mutate the dictionary used in other functions
    _params = params.copy() 

    # Look at parameters to generate kernel
    L = fit_params['L']
    rl_lambda = _params['rl_lambda']
    RL = np.linalg.inv(np.eye(len(L)) + (rl_lambda * L))
    _params.pop('rl_lambda', None)

    # Convert rank to int
    if 'rank' in _params:
        _params['rank'] = np.int(_params['rank'])
    kpmf_model = KPMF(**_params)
    kpmf_model.fit(X_train, X_val=X_val, F_kernel=RL)
    return kpmf_model

def kpmf_objective(X_train, X_val, fit_params):
    '''
    Return an objective function that that computes R2 of the trained
    PMF model on validation data given model hyperparameters
    '''
    assert fit_params is not None
    def obj(params):
        # Note that the parameter dictionary returned is specified *explicitly* with no
        # optional arguments. This makes reporting and retraining easy and unambiguous
        kpmf_model = train_kpmf(X_train, params, fit_params=fit_params, X_val=X_val, )
        r2_loss = kpmf_model.score(X_val)
        params = {**params, **kpmf_model.get_params()}
        params['max_iter'] = kpmf_model.iters_ran
        return {
            'loss': 1 - r2_loss,
            'status': hyperopt.STATUS_OK,
            'params': params,
        }
    return obj

def run_kpmf(X, fit_params, param_space,
             val_hidden_fraction,
             hidden_fraction, 
             n_repeats,
             hyperopt_iters = 10,
             seed=None,
             logistic=True):
    assert fit_params is not None
    log = get_logger()
    log.info('[Training KPMF model]')

    scaler = MCScaler(mode='0-1' if logistic else 'std')
    
    use_validation=True

    return run_mc_alg(X,
                      fold_objective= kpmf_objective,
                      retrain_model = train_kpmf,
                      fit_params = fit_params,
                      space=param_space,
                      scaler=scaler,
                      val_hidden_fraction=val_hidden_fraction,
                      hidden_fraction=hidden_fraction,
                      train_with_validation=use_validation,
                      n_repeats=n_repeats,
                      hyperopt_iters=hyperopt_iters,
                      hyperopt_seed=seed)

###############################################################################
#                   Functions to run NGMC
###############################################################################
def ngmc_param_space(args):
    param_space = {
        'lambda_f': hyperopt.hp.loguniform('lambda_f', 
                                            np.log(args.lambda_f_range[0]), 
                                            np.log(args.lambda_f_range[1])),
        'lambda_h': hyperopt.hp.loguniform('lambda_h', 
                                            np.log(args.lambda_h_range[0]), 
                                            np.log(args.lambda_h_range[1])),
        'alpha': hyperopt.hp.loguniform('alpha', 
                                            np.log(args.lr_range[0]), 
                                            np.log(args.lr_range[1])),
        'rank': hyperopt.hp.quniform('rank', 
                                     args.rank_range[0],
                                     args.rank_range[1],
                                     args.rank_step),
        'max_iter': hyperopt.hp.choice('max_iter', [args.max_iter])
    }

    if args.ppi is not None:
        param_space['lambda_p'] = hyperopt.hp.loguniform(
                                    'lambda_p', 
                                    np.log(args.lambda_p_range[0]), 
                                    np.log(args.lambda_p_range[1]))
        param_space['alpha_p'] = hyperopt.hp.loguniform(
                                    'alpha_p', 
                                    np.log(args.alpha_p_range[0]), 
                                    np.log(args.alpha_p_range[1]))

    return param_space

def train_ngmc(X_train, params, fit_params, X_val=None):
    _params = params.copy() 
    # Convert rank to int
    if 'rank' in _params:
        _params['rank'] = np.int(_params['rank'])
    
    ngmc_model = NGMC(**_params)
    ngmc_model.fit(X_train, X_val=X_val, P=fit_params['P'])
    return ngmc_model
    
def ngmc_objective(X_train, X_val, fit_params):
    '''
    Return an objective function that that computes R2 of the trained
    NGMC model on validation data given model hyperparameters
    '''
    def obj(params):
        # Note that the parameter dictionary returned is specified *explicitly* with no
        # optional arguments. This makes reporting and retraining easy and unambiguous
        ngmc_model = train_ngmc(X_train, params, fit_params, X_val=X_val)
        r2_loss = ngmc_model.score(X_val)
        params = {**params, **ngmc_model.get_params()}
        params['max_iter'] = ngmc_model.iters_ran
        return {
            'loss': 1 - r2_loss,
            'status': hyperopt.STATUS_OK,
            'params': params,
        }
    return obj

def run_ngmc(X, fit_params, param_space, 
            val_hidden_fraction,
            hidden_fraction,
            n_repeats,
            hyperopt_iters = 10,
            seed=None,
            logistic=True):
    assert logistic == True # Dummy argument so that calls to run_<alg> are the same

    log = get_logger()
    log.info('[Training NGMC model]')
    scaler = MCScaler(mode='0-1')
    use_validation=True

    return run_mc_alg(X,
                      fold_objective=ngmc_objective,
                      retrain_model = train_ngmc,
                      fit_params = fit_params,
                      space=param_space,
                      scaler=scaler,
                      val_hidden_fraction=val_hidden_fraction,
                      hidden_fraction=hidden_fraction,
                      train_with_validation=use_validation,
                      n_repeats=n_repeats,
                      hyperopt_iters=hyperopt_iters,
                      hyperopt_seed=seed)

def compute_training_curve(train_model, X_train, X_val, params, fit_params=None):
    '''
    Compute training curve given a function that that trains a model
    '''
    model = train_model(X_train, params, X_val=X_val, fit_params=fit_params)
    return dict(train=np.asarray(model.train_r2_hist),
                val  =np.asarray(model.val_r2_hist))

###############################################################################
# General functions to run MC algorithsm with HyperOpt
###############################################################################

def run_mc_alg( gis,
                fold_objective,
                retrain_model,
                space,
                scaler,
                val_hidden_fraction,
                hidden_fraction,
                fit_params = None,
                train_with_validation=False,
                n_repeats = 1,
                hyperopt_iters=3,
                hyperopt_seed=None):
    
    all_results = []
    all_params  = []
    all_models = []
    log = get_logger()
    param_search_training_curves = []
    hp_trials = []

    for i in range(n_repeats):
        log.info('[Outer fold: %i]' % i)
        X_train, X_test, test_mask = gi_train_test_split(gis, hidden_fraction)
        X_train = scaler.fit_transform(X_train)
        X_train_all = X_train.copy()
        if train_with_validation:
            gis['values'] = X_train_all
            log.info('- Holding out %f fraction of data for validation' % val_hidden_fraction)
            X_train, X_val, _ = gi_train_test_split(gis, val_hidden_fraction)
        log.info('- Performing hyperparameter search for %i iterations' % hyperopt_iters)
        trials = hyperopt.Trials()

        # NB: Ignore the returned hyperopt parameters, we want to know which parameters were
        # used even if they were default values for keyword arguments
        _ = hyperopt.fmin(fn=fold_objective(X_train, X_val, fit_params=fit_params), 
                                    space=space, 
                                    algo=hyperopt.tpe.suggest,
                                    max_evals = hyperopt_iters,
                                    trials=trials,
                                    show_progressbar=True,
                                    rstate=np.random.RandomState(hyperopt_seed))

        # NB: random state of hyperopt cannot be set globally, so we pass a 
        # np.RandomState object for reproducibility...
        hyperopt_seed += 1
        best_trial = trials.best_trial['result']

        # NB: that the parameter dictionary in trial['params'] is specified *explicitly* 
        # the retraining of new models then use *no optional arguments*. 
        # This makes reporting and retraining easy and unambiguous
        best_params = best_trial['params']

        # It's too fussy to serialize the models and save them as attachments in hyperopt.
        # Instead, we retrain a model and compute training curves instead.
        log.info('- Retraining model with validation data to get training curve')
        training_curve = compute_training_curve(retrain_model, X_train, X_val, best_params, 
                                                fit_params=fit_params)
        param_search_training_curves.append(training_curve)
        
        # Retrain model using the number of iterations and parameters found in hp search
        log.info('- Retraining model without validation to get best model')
        best_model = retrain_model(X_train_all, best_params, fit_params=fit_params)

        # Make predictions and evaluate the model
        X_fitted = best_model.X_fitted
        X_fitted = scaler.inverse_transform(X_fitted)

        if X_train.shape == X_train.T.shape and np.allclose(X_train, X_train.T, equal_nan=True):
            log.info('- Data was square, averaging predictions...')
            X_fitted = (X_fitted.T + X_fitted) / 2.
        #test_mask = ~np.isnan(X_test)

        #test_mask[np.tril_indices(len(test_mask))] = False

        results = evaluate_model(X_test[test_mask], X_fitted[test_mask])
        log.info('[Results for fold %i]' % i)
        log.info('- Best params for model')
        log_dict(log.info, best_params)
        log.info('- Results:')
        log_dict(log.info, results)
        
        hp_trials.append(trials.results)
        all_results.append(results)
        all_params.append(best_params)
        all_models.append(best_model)
    
    # Collate the results and return
    summarized, collected = summarize_results(all_results)
    return dict(summary=summarized, 
                fold_results=collected, 
                best_params=all_params), \
           all_models, \
           param_search_training_curves, \
           hp_trials

###############################################################################
#                           MAIN
###############################################################################
def main():
    # tf.enable_eager_execution()
    args = parse_args()
    setup_logging(args.logfile)

    log = get_logger()

    assert( 0 <= args.hidden_fraction <= 1 )
    
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)
    log.info('*' * 100)
    log.info('[Starting MC experiment]')
    log_dict(log.info, vars(args))
    log.info('[Loading input data]')
    obj = np.load(args.input_file)
    #check_gi_obj(obj)
    #mat = obj['values']

    # Set up experiments
    fit_params = None
    if args.mc_alg == 'PMF':
        param_space = pmf_param_space(args)
        run_experiment = run_pmf
    elif args.mc_alg == 'PMF_b':
        param_space = pmfb_param_space(args)
        run_experiment = run_pmfb
    elif args.mc_alg in ['KPMF', 'NGMC', 'KPMF_b']:
        # Experiments that need PPI network
        if args.ppi is not None:
            ppi = nx.read_edgelist(args.ppi)

        if args.mc_alg == 'KPMF':
            L = get_ppi_data(obj['rows'], ppi, mode='laplacian')
            param_space = kpmf_param_space(args)
            run_experiment = run_kpmf
            fit_params = dict(L=L)
        elif args.mc_alg == 'KPMF_b':
            L = get_ppi_data(obj['rows'], ppi, mode='laplacian')
            param_space = kpmfb_param_space(args)
            run_experiment = run_kpmfb
            fit_params = dict(L=L)
        elif args.mc_alg == 'NGMC':
            fit_params = dict(P=None)
            if args.ppi is not None:
                P = get_ppi_data(obj['rows'], ppi, mode='normalized_adjacency')
                fit_params['P'] = P
            param_space = ngmc_param_space(args)
            run_experiment = run_ngmc
        else:
            #TODO: implement experimental protocol for ngmc
            raise(NotImplementedError('{} option is invalid or not implemented'.format(args.mc_alg)))
                                    
    else:
        raise(NotImplementedError('{} option is invalid or not implemented'.format(args.mc_alg)))
    
    # Run experimental protocol
    results, models, training_curves, trials = \
        run_experiment(obj,
                        param_space = param_space,
                        fit_params = fit_params,
                        val_hidden_fraction=args.val_hidden_fraction,
                        hidden_fraction=args.hidden_fraction, 
                        n_repeats=args.n_repeats,
                        hyperopt_iters=args.n_hyperopt_iters,
                        seed=args.random_seed,
                        logistic=args.logistic)

    # Save results and other information
    log_results(results['summary'])
    with open(args.results_output, 'w') as f:
        json.dump(results, f, indent=2)
    joblib.dump(training_curves, args.training_curve_output)

    # TODO: save models the models cannot be pickled at the moment
    # We will need to implement a from dict and a to dict method
    joblib.dump(trials, args.models_output)

    joblib.dump(trials, args.trials_output)

# Run main and log and raise exceptions too
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log = get_logger()
        log.exception(str(e))
        raise e