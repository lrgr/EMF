import argparse

import sys
import os
sys.path.append(os.path.join("..", "src"))

import cloudpickle as cpkl

from benchmark_mf import evaluate_preds
from utils import summarize_results, log_results
from i_o import setup_logging, get_logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", required=True)
    return parser.parse_args()

def log_info(data, kind):
    log = get_logger()
    log.info("[Evaluating serialized {} models]".format(kind))
    log.info("- Serialized experimental data is a dictionary containing trained models,")
    log.info("  data used to train models, necessary information to reproduce evaluation procedures")
    log.info("- data indexed by: " + str(data.keys()))
    log.info("- data['fold_data'] contains train/test splits for folds...")
    log.info("  and has keys:" + str(data['fold_data'].keys()))
    log.info("- data['models_info'][<fold_num>] can be inspected for learned latent variables")
    log.info("  and has keys:" + str(data['models_info'][0].keys()))


def eval_EMF_data(model_file):

    with open(model_file, 'rb') as f:
        data = cpkl.load(f)

    log_info(data, 'EMF')

    imputed_Xs = data['imputed_Xs']
    test_Xs = data['fold_data']['test_Xs']
    masks = data['fold_data']['masks']

    results = evaluate_preds(test_Xs, imputed_Xs, masks)
    results, fold_results = summarize_results(results)
    log_results(results)

def main():
    args = parse_args()
    setup_logging()

    eval_EMF_data(args.model_file)

if __name__ == "__main__":
    main()