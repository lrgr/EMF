
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
    parser.add_argument('--source_gis', required=True)
    parser.add_argument('--target_gis', required=True)
    parser.add_argument('--swap', action='store_true')
    return parser.parse_args()

def load_cpkl(fp):
    with open (fp, 'rb') as f:
        return cpkl.load(f)

def main():
    args = parse_args()

    # tgt<->src blast scores (source on rows, tgt on cols)
    blast_scores = load_cpkl(args.input)

    src_genes = load_cpkl(args.source_gis)['cols']
    tgt_genes = load_cpkl(args.target_gis)['cols']
    # TRANSPOSE
    blast_rows = blast_scores['cols']
    blast_cols = blast_scores['rows']
    blast_values = blast_scores['values'].T
    ##
    
    print("Blast scores have shape:", blast_values.shape)
    print("# tgt genes:", len(tgt_genes))
    print("# src genes:", len(src_genes))

    blast_values[np.isnan(blast_values)] = 0.0

    blast_row_g2i = dict((g, i) for i,g in enumerate(blast_rows))
    blast_col_g2i = dict((g, i) for i,g in enumerate(blast_cols))
    tgt_g2i = dict((g,i) for i,g in enumerate(tgt_genes))
    src_g2i = dict((g,i) for i,g in enumerate(src_genes))

    tgt_idxs = [tgt_g2i[g] for g in tgt_genes if g in blast_rows]
    tgt_blast_idxs = [blast_row_g2i[g] for g in tgt_genes if g in blast_rows]
    src_idxs = [src_g2i[g] for g in src_genes if g in blast_cols]
    src_blast_idxs = [blast_col_g2i[g] for g in src_genes if g in blast_cols]

    print("Target genes with blast scores:", len(tgt_idxs))
    print("Soruce genes with blast scores:", len(src_idxs))

    extracted_scores = blast_values[tgt_blast_idxs]
    extracted_scores = extracted_scores[:, src_blast_idxs]

    tgt_restricted_scores = np.zeros((len(tgt_genes), len(blast_cols)))
    tgt_restricted_scores[tgt_idxs,:] = blast_values[tgt_blast_idxs]
    print(tgt_restricted_scores.shape)

    scores = np.zeros((len(tgt_genes), len(src_genes)))
    scores[:, src_idxs] = tgt_restricted_scores[:, src_blast_idxs]


    # Lots of sanity checks to make sure the reodering is done correctly
    assert(np.allclose(np.sum(scores), np.sum(extracted_scores)))
    assert(np.allclose(np.sum(scores, axis=0)[src_idxs], np.sum(extracted_scores, axis=0)))
    assert(np.allclose(np.sum(scores, axis=1)[tgt_idxs], np.sum(extracted_scores, axis=1)))
    assert(np.allclose(scores[scores > 0], extracted_scores[extracted_scores > 0]))

    test_tgt_genes = tgt_genes[np.argmax(scores, axis=0)]
    test_src_genes = src_genes[np.argmax(scores, axis=1)]

    n_checked = 0
    for t, s in zip(test_tgt_genes, test_src_genes):
        if t in blast_row_g2i and s in blast_col_g2i:
            blast_ti, blast_si = blast_row_g2i[t], blast_col_g2i[s]
            ti, si = tgt_g2i[t], src_g2i[s]
            assert(np.allclose(blast_values[blast_ti, blast_si], scores[ti, si]))
            n_checked += 1

    assert(n_checked > 0)

    print("Verified {} scores were correctly extracted".format(n_checked))
    print("Extracted shape: ", scores.shape)
    print("% > 0: ", np.sum(scores > 0) / (scores.shape[0] * scores.shape[1]))

    if args.swap:
        restricted_blast_data = dict(values=scores.T, rows=src_genes, cols=tgt_genes)
    else:
        restricted_blast_data = dict(values=scores, rows=tgt_genes, cols=src_genes)

    print("Saving extracted blast scores to:", args.output)
    
    with open (args.output, 'wb') as f:
        cpkl.dump(restricted_blast_data, f)

if __name__ == "__main__":
    main()
