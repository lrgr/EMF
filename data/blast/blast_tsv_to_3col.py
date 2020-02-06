import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.externals import joblib
import cloudpickle as cpkl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--thresh', required=True, type=float)
    parser.add_argument('--swap', action='store_true')
    return parser.parse_args()

# Constants for header names
_QSEQID = 'Q_ID'
_SSEQID = 'S_ID'
_BITSCORE = 'BITSCORE'
_RAWSCORE = 'RAWSCORE'
_EVALUE = 'EVALUE'

def main():
    args = parse_args()
    print('* loading df')
    col_names = [
       _QSEQID,
       _SSEQID,
       _BITSCORE,
       _RAWSCORE,
       _EVALUE]
    df = pd.read_csv(args.input, sep='\t', header=None, names=col_names)
    df.Q_ID = df.Q_ID.str.upper()
    df.S_ID = df.S_ID.str.upper()

    if args.swap:
        df = df[[_SSEQID, _QSEQID, _BITSCORE]]
    else:
        df = df[[_QSEQID, _SSEQID, _BITSCORE]]

    std = np.std(df[_BITSCORE])
    thresh = std * args.thresh
    print("* Thresholding {} x standard deviation at {}".format(args.thresh, thresh))

    df = df[df[_BITSCORE] > thresh]

    print("* # bitscores", len(df))

    df.to_csv(args.output, sep='\t', index=False)

if __name__ == "__main__":
    main()
