import hyperopt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

from cross_species_mf import (gene2index, read_homolog_list,
                              restrict_homs_to_gis)
from i_o import get_logger

###############################################################################
#                   Evaluation Utilities
###############################################################################

def nrmse(y_true, y_pred):
    '''
    Compute the 'Normalized Root Mean Squared Error metric as specified in 
    github.com/marinkaz/ngmc/example.py
    '''
    return np.sqrt(np.mean(np.multiply(y_true-y_pred, y_true-y_pred))/np.var(y_true))

def evaluate_model(y_true, y_pred):
    '''
    Compute the 'Normalized Root Mean Squared Error metric as specified in 
    github.com/marinkaz/ngmc/example.py
    '''

    # Default values to be inf or -inf if y_pred has NaNs
    results = {
        'pearson_R': (-np.inf, np.inf),
        'r2': np.inf,
        'mse': np.inf,
        'nrmse': np.inf,
    }

    if not np.all(np.isnan(y_pred)):
        pearsonr_score, pearsonr_pval = pearsonr(y_true, y_pred)
        results = {
            'pearsonr_score': pearsonr_score,
            'pearsonr_pval': pearsonr_pval,
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'nrmse': nrmse(y_true, y_pred)
        }
    return results

def collect_dict_items(dicts):
    '''
    Return dictionary of lists given lists of dictionaries with the same key
    '''
    collected = {}
    for k in dicts[0].keys():
        collected[k] = [d[k] for d in dicts]
    return collected

def summarize_results(result_dicts):
    '''
    Computes mean, std, min, max for a list of dictinaries of results.
    This function returns both summary statistics and converts list of dicts
    into dicts of lists for convenience
    '''
    collected = collect_dict_items(result_dicts)
    summarized = {}
    for k, vals in collected.items():
        summarized[k] = {'mean':np.mean(vals),
                         'std':np.std(vals),
                         'min':np.min(vals),
                         'max':np.max(vals),}
    return summarized, collected

def log_results(summarized_results, ntabs=1):
    log = get_logger()
    df = pd.DataFrame(summarized_results).T
    df = df[['mean', 'std', 'min', 'max']]
    log.info('[Results]')
    lines = df.to_string().split('\n')
    for s in lines: log.info('\t' * ntabs + s) 

###############################################################################
#                   Other Utilities
###############################################################################

def sparsity(X, percent=True):
    '''
    Compute the sparsity (proportion of NaNs) in given nparray
    '''
    s = np.sum(np.isnan(X)) / X.size
    if percent:
        s *= 100.
    return s

def check_gi_obj(obj):
    '''
    Perform sanity check for obj containing GI data: Check that rows == cols
    and that the matrix is square
    '''
    mat = obj['values']
    rows = obj['rows']
    cols = obj['cols']

    assert len(rows) == len(cols), \
        'Matrix is not square and has shape {}'.format((len(rows), len(cols)))
    assert len(set(rows) & set(cols)) == len(rows), \
        'Gene names on row and cols are not unique (# names: {}, shape: {}).'.format(
            len(set(rows) & set(cols)), mat.shape)
    assert all(rows == cols)

def get_ppi_data(genes, ppi, mode='laplacian', eps=1e-15):
    '''
    Compute gene-by-gene matrix from ppi with respect to given genes.
    Modes:
        - 'laplacian' returns the laplacian
        - 'normalized_adjacency' returns adjacency matrix with rows
           normalized to sum to 1.
    '''

    log = get_logger()
    
    ppi_genes = set(ppi.nodes())
    missing_genes = [g for g in genes if g not in ppi_genes]

    log.info('- Adding %d missing genes to ppi', len(missing_genes))
    ppi.add_nodes_from(missing_genes)

    ppi = ppi.subgraph(genes)
    if (not nx.is_connected(ppi)):
        cc_list = list(nx.connected_components(ppi))
        cc_sizes = [len(x) for x in cc_list]
        log.warning('- Network has %d connected components', len(cc_list))
        log.warning('\t- sizes: {}'.format(cc_sizes))

    if mode == 'laplacian':
        return nx.laplacian_matrix(ppi, nodelist=genes).todense()
    elif mode == 'normalized_adjacency':
        A = nx.adjacency_matrix(ppi, nodelist=genes).todense()
        return A / (A.sum(axis=0) + eps)
    else:
        raise NotImplementedError

def get_laplacian(genes, ppi_path):
    with open(ppi_path, 'r') as f:
        ppi = nx.read_edgelist(ppi_path)
        L = get_ppi_data(genes, ppi, mode='laplacian')
    return L

###############################################################################
#                           hyperopt
###############################################################################

def add_hyperopt_loguniform(space, name, minmax):
    assert name not in space
    space[name] = hyperopt.hp.loguniform(name, np.log(minmax[0]), np.log(minmax[1]))
    return space

def add_hyperopt_quniform(space, name, minmax, step):
    assert name not in space
    space[name] = hyperopt.hp.quniform(name, minmax[0], minmax[1], step)
    return space


###############################################################################
#                           Loading things verbosely...
###############################################################################

def safe_verbose_load_gis(fp, to_upper=True):
    log = get_logger()
    log.info('* loading GIs from {}'.format(fp))
    obj = np.load(fp)
    check_gi_obj(obj) # TODO: move this to a utils file
    X = obj['values']
    log.info('* GIs have shape {}'.format(X.shape))
    log.info('* GIs are {:.3f}% sparse'.format(sparsity(X)))
    if to_upper:
        log.info('Converting genes names in GIs to use uppercase')
        genes = obj['rows']
        genes = np.asarray([g.upper() for g in genes])
    return genes, X

def safe_verbose_load_homologs(fp, tgt_genes, src_genes, to_upper=True):
    log = get_logger()
    log.info('* loading homologs from {}'.format(fp))

    if to_upper:
        log.info('* Converting genes in homologs to use uppercase')
    homologs = read_homolog_list(fp, to_upper=to_upper)
    homologs = restrict_homs_to_gis(homologs, tgt_genes, src_genes)

    tgt_g2i = gene2index(tgt_genes)
    src_g2i = gene2index(src_genes)
    hom_idxs = np.asarray([(tgt_g2i[t], src_g2i[s]) for t, s in homologs])
    
    log.info('* found {} homolog pairs w.r.t {} target and {} source genes'.format(
             len(hom_idxs), len(tgt_genes), len(src_genes)))

    return hom_idxs
