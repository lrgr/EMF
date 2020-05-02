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

def get_eval_pair_list(pairs, row_g2i, col_g2i, gi_data):
    values = gi_data['values']

    pairlist_1 = []
    pairlist_2 = []
    for A, B in pairs:
        A_r = row_g2i.get(A)
        B_c = col_g2i.get(B)
        A_c = col_g2i.get(A)
        B_r = row_g2i.get(B)
        if (A_r is not None) and \
               (B_c is not None) and \
               (A_c is not None) and \
               (B_r is not None):

            v_ab = values[A_r, B_c]
            v_ba = values[B_r, A_c]

            if not (np.isnan(v_ab) or np.isnan(v_ba)):
                pairlist_1.append((A_r, B_c))
                pairlist_2.append((B_r, A_c))
            else:
                pass

        elif (A_r is not None) and \
                (B_c is not None):
            if not np.isnan(values[A_r, B_c]):
                pairlist_1.append((A_r, B_c))
                pairlist_2.append((A_r, B_c))
            else:
                pass

        elif (A_c is not None) and \
                (B_r is not None):
            if not np.isnan(values[B_r, A_c]):
                pairlist_1.append((B_r, A_c))
                pairlist_2.append((B_r, A_c))
            else:
                pass
        else:
            continue
    
    pairlist_1 = tuple(zip(*pairlist_1))
    pairlist_2 = tuple(zip(*pairlist_2))
    return pairlist_1, pairlist_2
    
def gi_train_test_split_w_pairlists(gi_data, hf):
    '''
    Sample train/test set but return lists of indices whose indexed values should be
    averaged for evaluation
        [(A,B), ...], [(B,A),...]
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
    
    # This implements train/test over *all* possible pairs,
    # in expectation is equivalent to CV over observed pairs

    value_mask = ~np.isnan(values)
    test_mask =  np.logical_and(value_mask,  test_mask)
    train_mask = np.logical_and(value_mask, ~test_mask)
    
    train_X = np.where(train_mask, values, np.nan)
    test_X = np.where(test_mask, values, np.nan)
    
    eval_pairs1, eval_pairs2 = get_eval_pair_list(test_pairs, row_g2i, col_g2i, gi_data)

    assert(np.all(~np.isnan(test_X[test_mask])))

    assert(np.all(~np.isnan(test_X[eval_pairs1[0], eval_pairs1[1]])))
    assert(np.all(~np.isnan(test_X[eval_pairs2[0], eval_pairs2[1]])))

    return train_X, test_X, (eval_pairs1, eval_pairs2)

def sym_train_test_split(gi_data, hf):
    values = gi_data['values']
    
    assert(np.allclose(values, values.T, equal_nan=True))
    N = len(values)
    
    eval_mask = np.random.uniform(size=values.shape) < hf
    
    lower_tri_mask = np.tri(N, N, k=-1, dtype=bool)
    
    eval_mask = np.logical_and(eval_mask, lower_tri_mask)
    eval_mask = np.logical_and(eval_mask, ~np.isnan(values))
        
    diag = np.zeros(len(values), dtype=bool)
    
    test_mask = np.logical_or(eval_mask, eval_mask.T)
    
    train_mask = ~test_mask
    train_mask[diag] = False
    
    train_X = np.where(train_mask, values, np.nan)
    test_X = np.where(test_mask, values, np.nan)
    
    assert(np.sum(eval_mask) / np.sum(~np.isnan(test_X)) == 0.5)

    return train_X, test_X, eval_mask

def _is_sym(gi_data):

    if len(gi_data['rows']) != len(gi_data['cols']):
        return False
        
    cond0 = np.allclose(gi_data['values'], gi_data['values'].T, equal_nan=True)
    cond1 = np.all(gi_data['rows'] == gi_data['cols'])
    return cond0 and cond1

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
    
    # This implements train/test over *all* possible pairs,
    # in expectation is equivalent to CV over observed pairs

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