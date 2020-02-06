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

    # print('* computing set')
    # A_scores_only_geneset = set(df['A'].values)
    # B_scores_only_geneset = set(df['B'].values)

    # A_genes = np.asarray(list(A_scores_only_geneset))
    # B_genes = np.asarray(list(B_scores_only_geneset))

    # A_n2i = dict((n, i) for i, n in enumerate(A_genes))
    # B_n2i = dict((n, i) for i, n in enumerate(B_genes))

    # A_idxs = [A_n2i[n] for n in df['A'].values]
    # B_idxs = [B_n2i[n] for n in df['B'].values]
    # scores = df['score'].values
    # print('* making dense matrix')
    # X = csr_matrix((scores, (A_idxs, B_idxs))).toarray()

    # print('* loading ppis to restrict dense matrix to ppi nodes')
    # A_ppi_only_geneset = get_ppi_nodes(args.A_ppi)
    # B_ppi_only_geneset = get_ppi_nodes(args.B_ppi)

    # print('* Taking set intersections to get genes in ppi with scores')
    # A_genes = list(A_ppi_only_geneset & A_scores_only_geneset)
    # B_genes = list(B_ppi_only_geneset & B_scores_only_geneset)

    # print('A:', len(A_genes))
    # print('B:', len(B_genes))

    # A_idxs = [A_n2i[n] for n in A_genes]
    # B_idxs = [B_n2i[n] for n in B_genes]

    # print('* restricting and saving')
    # X = X[A_idxs, :][:, B_idxs]
    # print(X.shape)
    # joblib.dump(dict(X=X, A_nodes=A_genes,
    #                  B_nodes=B_genes), args.output)

if __name__ == "__main__":
    main()
