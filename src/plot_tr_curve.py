import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.externals import joblib


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    
    # Main parser (currently assuming we only do gradient descent)
    parser.add_argument('-cs', '--training_curves',  type=str, nargs='+', required=True)
    parser.add_argument('-n', '--names',  type=str, nargs='+', required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-t', '--title', type=str, required=True)
    parser.add_argument('--x_min', type=int, required=False, default=0)
    parser.add_argument('--y_min', type=float, required=False, default=None)
    parser.add_argument('--y_max', type=int, required=False, default=None)
    parser.add_argument('--x_max', type=int, required=False, default=None)
    

    return parser.parse_args()

def main():
    colors = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
    n_colors = len(colors)
    args = parse_args()
    assert len(args.names) == len(args.training_curves)
    training_curves = {name: joblib.load(fp) 
                        for name, fp in zip(args.names, args.training_curves)}
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    lines = []
    for i, fp in enumerate(args.training_curves):
        curves = joblib.load(fp)
        for c in curves:
            line = plot_tr_curve(ax, c['train'], c['val'], colors[i % n_colors]) 
        lines.append(line)

    val_legend_line = plt.plot([],[], 'k-')[0]
    tr_legend_line = plt.plot([],[], 'k--')[0]

    ax.set_ylim(top=1)
    ax.set_xlim(left=0)

    if args.y_min is not None:
        ax.set_ylim(bottom=args.y_min)
    if args.y_max is not None:
        ax.set_ylim(top=args.y_max)
    if args.x_max is not None:
        ax.set_xlim(left=args.x_min)
    if args.x_max is not None:
        ax.set_xlim(right=args.x_max)
    
    ax.legend(lines + [val_legend_line, tr_legend_line], args.names + ['(Training)', '(Validation)'], fontsize=15)
    ax.set_xlabel('Iterations', fontsize=20)
    ax.set_ylabel('R2 score', fontsize=20)
    ax.set_title(args.title.capitalize(), fontsize=30)


    plt.tight_layout()
    plt.savefig(args.output)

def plot_tr_curve(ax, tr, val, color):
    assert(len(tr) == len(val))
    n_iters = len(tr)
    line = ax.plot(np.arange(n_iters), tr, color=color, linestyle='-')[0]
    ax.plot(np.arange(n_iters), val, color=color, linestyle='--')

    return line

if __name__ == "__main__":
    main()
