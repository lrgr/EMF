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
    return parser.parse_args()

# Constants for header names
_QSEQID = 'Q_ID'
_SSEQID = 'S_ID'
_BITSCORE = 'values'
_RAWSCORE = 'RAWSCORE'
_EVALUE = 'EVALUE'

def main():
    '''
    Create dense matrix of scores from TSV of BLAST scores.

    cloudpickled data dictionary has keys:
        - rows : 1D array of gene names
        - cols : 1D array of gene nanes
        - {values, RAWSCORE, EVALUE} : (n_rows, n_cols) 2D array of bitscores, 
            raw bitscores, and e-values respectively
    '''
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
    print(df.head())

    # Turn (gene1, gene2)->[values] into mult-index / 2D matrix of values
    print("\t- pivoting dataframe...")
    df = df.pivot(index=_QSEQID, columns=_SSEQID)

    data = {
        'rows': df[_BITSCORE].index.values.astype(str),
        'cols': df[_BITSCORE].columns.values.astype(str)
    }
    for val_type in [_BITSCORE, _RAWSCORE, _EVALUE]:
        print("* Getting {}".format(val_type))
        val_array = df[val_type].values
        print("\t- Shape:", val_array.shape)
        data[val_type] = val_array

    print("* Saving to ", args.output)
    with open(args.output, 'wb') as f:
        cpkl.dump(data, f)

if __name__ == "__main__":
    main()
