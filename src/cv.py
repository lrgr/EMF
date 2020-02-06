import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split

def get_mask(pairs, shape, row_g2i, col_g2i, sym=True):
    '''
    Convert a list of pairs, into a boolean indicator matrix, m, where
    (a,b) in pairs, is indexed into m[ai, bj] and m[bi, aj] where possible.
    
    if sym = False, then a pair is indicated only once so either
        m[ai, bj] == True xor m[bi, aj] == True
    '''
    mask = np.zeros(shape, dtype=bool)
    for i, (a, b) in enumerate(pairs):
        inserted=False
        if a in row_g2i and b in col_g2i:
            mask[row_g2i[a], col_g2i[b]] = True
            inserted = True
        
        if not sym and inserted:
            assert(inserted)
            continue

        if a in col_g2i and b in row_g2i:
            mask[row_g2i[b], col_g2i[a]] = True
            inserted = True
        assert(inserted)
    return mask
    
def gi_train_test_split(gi_data, hf):
    '''
    Returns Train_X, Test_X, eval_mask
    
    eval_mask is mask of unique pairs for evaluation
    '''

    rows = gi_data['rows']
    cols = gi_data['cols']
    values = gi_data['values']
    col_g2i = dict((n, i) for i, n in enumerate(cols))
    row_g2i = dict((n, i) for i, n in enumerate(rows))
    
    rowset = set(rows)
    colset = set(cols)
    
    pairs = product(rows, cols)
    pairs =  set(frozenset((a,b)) for a,b in pairs if a != b)
    pairs = [tuple(p) for p in pairs]
    
    train_pairs, test_pairs = train_test_split(pairs, test_size=hf)
    test_mask = get_mask(test_pairs, values.shape, row_g2i, col_g2i)
    
    # TODO: This implements train/test over *all* possible pairs, but this needs fixing... 
    # since not all pairs have value in GI matrix
    value_mask = ~np.isnan(values)
    test_mask =  np.logical_and(value_mask,  test_mask)
    train_mask = np.logical_and(value_mask, ~test_mask)
    
    train_X = np.where(train_mask, values, np.nan)
    test_X = np.where(test_mask, values, np.nan)
    
    # Get mask for evaluation time... (this mask only has one gi score per pair)
    eval_mask = get_mask(test_pairs, values.shape, row_g2i, col_g2i, sym=False)
    eval_mask = np.logical_and(value_mask, eval_mask)
    assert(np.all(~np.isnan(test_X[test_mask])))
    assert(np.all(~np.isnan(test_X[eval_mask])))
    return train_X, test_X, eval_mask