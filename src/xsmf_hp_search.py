import json

import cloudpickle as cpkl
import hyperopt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score

from cv import gi_train_test_split
from i_o import get_logger, log_dict, setup_logging
from matrix_completion import MCScaler
from utils import (add_hyperopt_loguniform, add_hyperopt_quniform,
                   evaluate_model, log_results, sparsity, summarize_results)
from xsmf import XSMF, normalize_sim_scores


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    
    # Parser
    parser.add_argument('-tgt', '--target_gis', type=str, required=True)
    parser.add_argument('-src', '--source_gis', type=str, required=True)
    parser.add_argument('-sims', '--sim_scores', type=str, required=True)
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

    # Model hyper-parameters to search over
    parser.add_argument('--lambda_sim', type=float, nargs=2, required=True)
    parser.add_argument('--lambda_source', type=float, nargs=2, required=True)
    parser.add_argument('--lambda_u', type=float, nargs=2, required=True)
    parser.add_argument('--lambda_v', type=float, nargs=2, required=True)
    parser.add_argument('--lambda_us', type=float, nargs=2, required=True)
    parser.add_argument('--lambda_vs', type=float, nargs=2, required=True)
    parser.add_argument('--rank', type=int, nargs=2, required=True)
    parser.add_argument('--rank_step', type=int, required=True)
    parser.add_argument('--lr', type=float, nargs=2, required=True)
    
    # Fixed parameters
    parser.add_argument('--max_iter', type=int, required=True)
    parser.add_argument('--report_every', type=int, required=True)
    parser.add_argument('--early_stop', type=int, required=True)

    args = parser.parse_args()

    return args


###############################################################################
#                           MAIN
###############################################################################

def xsmf_param_space(args):
    param_space = {}
    add_hyperopt_loguniform(param_space, 'lambda_sim', args.lambda_sim)
    add_hyperopt_loguniform(param_space, 'lambda_src', args.lambda_source)
    add_hyperopt_loguniform(param_space, 'lambda_u', args.lambda_u)
    add_hyperopt_loguniform(param_space, 'lambda_v', args.lambda_v)
    add_hyperopt_loguniform(param_space, 'lambda_us', args.lambda_us)
    add_hyperopt_loguniform(param_space, 'lambda_vs', args.lambda_vs)
    add_hyperopt_loguniform(param_space, 'lr', args.lr)
    add_hyperopt_quniform(param_space, 'rank', args.rank, args.rank_step)

    param_space['max_iter'] = args.max_iter
    param_space['early_stop'] = args.early_stop
    param_space['report_every'] = args.report_every
    return param_space

def get_xsmf_obj(X_train, X_val, src_X, sim_scores):
    def objective(params):
        # Note that the parameter dictionary returned is specified *explicitly* with no
        # optional arguments. This makes reporting and retraining easy and unambiguous
        print(params)
        model = XSMF(X_tgt=X_train, X_val=X_val, X_src=src_X, 
                      sim_scores=sim_scores,
                        **params)
        model.fit()
        
        r2_loss = model.score(X_val)
        params = {**params, **model.get_params()} # store all parameters explicitly per trial
        params['max_iter'] = model.iters_ran
        tf.reset_default_graph()

        return {
            'loss': 1 - r2_loss,
            'status': hyperopt.STATUS_OK,
            'params': params,
        }
    return objective

def run_xsmf_experiment(tgt_gis, src_gis, sim_scores, space, val_hf, test_hf, 
                        n_repeats, hp_iters, hp_seed,
                        L_tgt=None, L_src=None):
    all_results = []
    all_params  = []
    all_models = []
    log = get_logger()
    param_search_training_curves = []
    hp_trials = []

    src_X_scaled = MCScaler(mode='std').fit_transform(src_gis['values'])
    for i in range(n_repeats):
        log.info('[Outer fold: %i]' % i)
        scaler = MCScaler(mode='std')
        X_train, X_test, eval_mask = gi_train_test_split(tgt_gis, test_hf)
        X_train = scaler.fit_transform(X_train)
        X_train_all = X_train.copy()
        
        tgt_gis['values'] = X_train
        log.info('- Holding out %.3f fraction of data for validation' % val_hf)
        X_train, X_val, _ = gi_train_test_split(tgt_gis, val_hf)
        
        log.info('- Performing hyperparameter search for %i iterations' % hp_iters)
        trials = hyperopt.Trials()
        # NB: Ignore the returned hyperopt parameters, we want to know which parameters were
        # used even if they were default values for keyword arguments
        _ = hyperopt.fmin(fn=get_xsmf_obj(X_train, X_val, src_X_scaled, sim_scores), 
                        space=space, 
                        algo=hyperopt.tpe.suggest,
                        max_evals = hp_iters,
                        trials=trials,
                        show_progressbar=True,
                        rstate=np.random.RandomState(hp_seed))
        # NB: random state of hyperopt cannot be set globally, so we pass a 
        # np.RandomState object for reproducibility...
        hp_seed += 1
        best_trial = trials.best_trial['result']

        # NB: that the parameter dictionary in trial['params'] is specified *explicitly* 
        # the retraining of new models then use *no optional arguments*. 
        # This makes reporting and retraining easy and unambiguous
        best_params = best_trial['params']

        # TODO:
        # # It's too fussy to serialize the models and save them as attachments in hyperopt.
        # # Instead, we retrain a model and compute training curves instead.
        # log.info('- Retraining model with validation data to get training curve')
        # training_curve = compute_training_curve(retrain_model, X_train, X_val, best_params, 
        #                                         fit_params=fit_params)
        # param_search_training_curves.append(training_curve)

        # Retrain model using the number of iterations and parameters found in hp search
        log.info('- Retraining model without validation to get best model')
        best_model = XSMF(X_tgt=X_train_all, X_val=None, X_src=src_X_scaled,  
                            sim_scores=sim_scores,
                            **best_params)
       
        best_model.fit()

        # Make predictions and evaluate the model
        X_fitted = best_model.X_fitted
        X_fitted = scaler.inverse_transform(X_fitted)

        if len(tgt_gis['rows']) == len(tgt_gis['cols']) and np.all(tgt_gis['rows'] == tgt_gis['cols']):
            log.info('* Averaging over pairs because input is symmetric')
            X_fitted = (X_fitted.T + X_fitted) / 2.
        
        # test_mask = ~np.isnan(X_test)

        # test_mask[np.tril_indices(len(test_mask))] = False

        results = evaluate_model(X_test[eval_mask], X_fitted[eval_mask])
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

def main():
    args = parse_args()
    setup_logging(args.logfile)

    log = get_logger()

    assert( 0 <= args.hidden_fraction <= 1 )
    
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)
    log.info('*' * 100)
    log.info('[Starting MC experiment]')
    log_dict(log.info, vars(args))
    log.info('[Loading target GIs]')
    with open(args.target_gis, 'rb') as f:
        tgt_gis = cpkl.load(f)
    
    log.info('[Loading source GIs]')
    with open(args.source_gis, 'rb') as f:
        src_gis = cpkl.load(f)

    
    log.info('[Loading sim scores]')
    with open(args.sim_scores, 'rb') as f:
        sim_scores_data = cpkl.load(f)
    sim_scores = sim_scores_data['values']
    sim_scores = sim_scores / np.max(sim_scores) # Normalize

    # log.info('\t- %d scores', len(sim_scores))
    
    hp_param_space = xsmf_param_space(args)

    results, models, training_curves, trials = \
        run_xsmf_experiment(tgt_gis=tgt_gis,
                            src_gis=src_gis,
                            space=hp_param_space,
                            sim_scores=sim_scores,
                            val_hf=args.val_hidden_fraction,
                            test_hf=args.hidden_fraction, 
                            n_repeats=args.n_repeats,
                            hp_iters=args.n_hyperopt_iters,
                            hp_seed=args.random_seed)
    # Save results and other information
    log_results(results['summary'])
    with open(args.results_output, 'w') as f:
        json.dump(results, f, indent=2)

    with open(args.training_curve_output, 'wb') as f:
        cpkl.dump(training_curves, f)

    # TODO: save models the models cannot be pickled at the moment
    # We will need to implement a from dict and a to dict method
    with open(args.models_output, 'wb') as f:
        cpkl.dump(trials, f)

    with open(args.trials_output, 'wb') as f:
        cpkl.dump(trials, f)



if __name__ == "__main__":
    main()
