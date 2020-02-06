import logging

import numpy as np
import tensorflow as tf
from scipy.special import expit
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score

from i_o import get_logger, setup_logging
from matrix_completion import mc_train_test_split

class XSMF(BaseEstimator):
    '''
    Cross-Species Matrix Factorization Model (XSMF)

    Parameters
    ----------
    X: 
        Target matrix
    X_source: 
        Source matrix of related species
    hom_idxs: list(tuples)
        List of (target, source) tuples of homologs 
    '''

    def __init__(self, X, X_source, hom_idxs,
                 X_val=None,
                 lambda_hom=0.1,
                 lambda_source=0.1,
                 lambda_u=0.1,
                 lambda_v=0.1,
                 lambda_us=0.1,
                 lambda_vs=0.1,
                 rank=40,
                 max_iter=150,
                 lr=0.1,
                 report_every=-1,
                 verbosity=logging.INFO,
                 early_stop=10,
                 logistic=False):
        self.log = get_logger(verbosity)
        self.X = X
        self.X_source = X_source
        self.X_val = X_val
        self.hom_idxs = hom_idxs
        self.logistic = logistic

        #check_square(X)
        check_hom_idxs(self.X, self.X_source, self.hom_idxs)
        check_1to1_homs(self.hom_idxs)

        self.lambda_hom = lambda_hom # reg param for shared factors
        self.lambda_source = lambda_source #reg param for source non-homs
        self.lambda_u = lambda_u # reg param for target U
        self.lambda_v = lambda_v # reg param for target V
        self.lambda_us = lambda_us # reg param for source U
        self.lambda_vs = lambda_vs # reg param for target V

        self.rank = int(rank)
        self.max_iter = max_iter
        self.lr = lr
        self.report_every = report_every
        self.verbosity = verbosity
        self.early_stop = early_stop

        # Some book-keeping for homologs
        self.tgt_hom_idxs = self.hom_idxs[:, 0]
        self.src_hom_idxs = self.hom_idxs[:, 1]
        self.n_homs = len(hom_idxs)

        # Initialize source and target latent factors
        self._initialize_factors()

        self.src_mask = ~np.isnan(self.X_source)
        self.tgt_mask = ~np.isnan(self.X)

        if self.X_val is not None:
            self.val_mask = ~np.isnan(self.X_val)

        # Initialize tensorflow variables
        self._initialize_tf_vars()

        self.X_fitted = None
        self.XS_fitted = None

        # History
        self.iters_ran = -1
        self.train_loss_hist = []
        self.train_r2_hist = []
        
        if self.X_val is not None:
            self.val_r2_hist = []
    
    def _initialize_factors(self):
        # Initialize target latent factors U, V
        self.N, self.M = self.X.shape
        assert self.M > self.rank
        assert self.N > self.rank

        # Initialize source latent factors US, VS
        self.NS, self.MS = self.X_source.shape
        assert self.MS > self.rank
        assert self.NS > self.rank

        self.U = np.random.randn(self.N, self.rank).astype(np.float64)
        self.V = np.random.randn(self.M, self.rank).astype(np.float64)
        self.US = np.random.randn(self.NS, self.rank).astype(np.float64)
        self.VS = np.random.randn(self.MS, self.rank).astype(np.float64)
    
    def _initialize_tf_vars(self):
        '''
        Initialize internal corresponding internal tensorflow representation of 
        factors and data
        '''
        self._tgt_reordering = selected_idxs_first_ordering(self.tgt_hom_idxs, self.N) 
        self._src_reordering = selected_idxs_first_ordering(self.src_hom_idxs, self.NS)

        XT = reorder_matrix_rows(self._tgt_reordering, self.X, mode='to')
        XS = reorder_matrix_rows(self._src_reordering, self.X_source, mode='to')

        if self.X_val is not None:
            X_val = reorder_matrix_rows(self._tgt_reordering, self.X_val, mode='to')
            self._X_val  = tf.constant(X_val)

        self._XT  = tf.constant(XT)
        self._XS = tf.constant(XS)

        init = tf.initializers.random_normal()
        self._UT = tf.get_variable('UT', shape=(self.N - self.n_homs, self.rank),
                                    initializer=init, dtype=tf.float64)
        self._UH = tf.get_variable('UH', shape=(self.n_homs, self.rank),
                                    initializer=init, dtype=tf.float64)
        self._US = tf.get_variable('US', shape=(self.NS - self.n_homs, self.rank),
                                    initializer=init, dtype=tf.float64)
        self._VT = tf.get_variable('VT', shape=(self.M, self.rank),
                                    initializer=init, dtype=tf.float64)
        self._VS = tf.get_variable('VS', shape=(self.MS, self.rank),
                                    initializer=init, dtype=tf.float64)

    def fit(self):
        '''
        --- outside of fit ---
        factors in as numpy matrices ...
        --- in fit ---
        0) whenever fit is called, reorder factors to make loss easy to compute
        1) book-keep ordering to put all homologs together in 
           primary submatrices of left factors
           - Call reordered GI matrices X ---> R, X_source ---> R_source
        2a) Compute H.Vt, Ut.Vt, H.Us, Ut.Us. separately
        2b) iteratate: compute loss + backprop w.r.t R and R_source
        2c) 
        
        +--+   
        |H |
        +--+   x V = [HxVt | UtxVt]^T
        |Ut|
        +--+
        '''
        # NB: refitting with more iterations is not implemented 
        # We will use H, Ut, Us, Vt, Vs to denote re-ordered matrices
        # Initialize target and source latent factors:

        # Re-order so that homologs are in principle submatrices
        
        loss = self.loss()
        train_r2_tf = self.tgt_r2_score(self._XT)

        if self.X_val is not None:
            val_r2_tf = self.tgt_r2_score(self._X_val)
        
        _early_stopped = False
        with tf.Session() as sess:
            #optim = tf.train.GradientDescentOptimizer(self.lr)
            optim = tf.train.AdamOptimizer(learning_rate=self.lr)
            train_step = optim.minimize(loss)

            sess.run(tf.initializers.global_variables())
            for i in range(self.max_iter):
                # Training step
                err, train_r2 = sess.run([loss, train_r2_tf])
                self.train_loss_hist.append(err)
                self.train_r2_hist.append(train_r2)

                if self.X_val is not None:
                    val_r2 = sess.run(val_r2_tf)
                    self.val_r2_hist.append(val_r2)
                sess.run(train_step)

                # Reporting
                if self.report_every >= 1 and i % self.report_every == 0:
                    self.log.info("\titer={:5d} | Training   | Cost={:8.4f}".format(i, err))
                    self.log.info("\titer={:5d} | Training   | R2={:8.4f}".format(i, train_r2))
            
                    if self.X_val is not None:
                         self.log.info("\titer={:5d} | Validation | R2={:8.4f}".format(i, val_r2))
                
                    if np.isnan(err):
                        self.log.warn('\tquitting training early due to NaN loss')
                        break

                # Early stop if validation loss stops decreasing
                if self.X_val is not None and self.early_stop > 0:
                    if self.stop_iterating():
                        self.log.info("\tstopping early due to non-decreasing validation loss")
                        _early_stopped = True
                        break

            if not _early_stopped:
                self.log.info('\tstopping due to max-iterations reached')
            # Convert and re-order fitted tensorflow factors back to nparrays
            Uh, Ut, Us, Vt, Vs = \
                sess.run([self._UH, self._UT, self._US, self._VT, self._VS])
            
        Ut = np.concatenate([Uh, Ut])
        Us = np.concatenate([Uh, Us])
        self.XS_fitted = np.dot(Us, Vs.T)
        self.X_fitted = np.dot(Ut, Vt.T)

        if self.logistic:
            self.XS_fitted = expit(self.XS_fitted)
            self.X_fitted = expit(self.X_fitted)

        # TODO: self.U, self.US, self.V, self.VS need to be reordered
        
        self.U = reorder_matrix_rows(self._tgt_reordering, Ut, mode='from')
        #self.V = reorder_matrix_rows(self._tgt_reordering, Vt, mode='from')
        self.V = Vt
        self.US = reorder_matrix_rows(self._src_reordering, Us, mode='from')
        self.VS = Vs
        #self.VS = reorder_matrix_rows(self._src_reordering, Vs, mode='from')

        self.XS_fitted = reorder_matrix_rows(self._src_reordering, self.XS_fitted, mode='from')
        self.X_fitted = reorder_matrix_rows(self._tgt_reordering, self.X_fitted, mode='from')

        assert np.allclose(self.X_fitted, np.dot(self.U, self.V.T))
        assert np.allclose(self.XS_fitted, np.dot(self.US, self.VS.T))


        # Store how many iterations we ran for for convenience
        self.iters_ran = len(self.train_loss_hist)

        return self
    
    def loss(self):
        '''
        Returns tf loss variable
        '''
        return self._loss_helper(Xt=self._XT, 
                                 Xs=self._XS,
                                 Ut=self._UT,
                                 Uh=self._UH,
                                 Us=self._US,
                                 Vt=self._VT,
                                 Vs=self._VS,
                                 lam_homs=self.lambda_hom,
                                 lam_src=self.lambda_source,
                                 lam_u=self.lambda_u, 
                                 lam_v=self.lambda_v,
                                 lam_us=self.lambda_us, 
                                lam_vs=self.lambda_vs,
                                logistic=self.logistic)
    @staticmethod
    def _loss_helper(Xt, Xs, Ut, Uh, Us,  Vt, Vs,
             lam_homs, lam_src, lam_u, lam_v, lam_us, lam_vs,
             logistic):

        '''
        Returns
        -------
        TF variables 
        Parameters
        ----------
        Xt : array-like
        Xs : array-like
        Ut : array-like
        Vs : array-like
        homs : List of tuples
        lambda_u : float
        lambda_v : float
        lambda_us : float
        lambda_vs : float
        lambda_homs : float
        lambda_src : float
        '''
        UhVt = tf.matmul(Uh, Vt, transpose_b=True)
        UtVt = tf.matmul(Ut, Vt, transpose_b=True)
        UhVs = tf.matmul(Uh, Vs, transpose_b=True)
        UsVs = tf.matmul(Us, Vs, transpose_b=True)

        if logistic:
            UhVt = tf.sigmoid(UhVt)
            UtVt = tf.sigmoid(UtVt)
            UhVs = tf.sigmoid(UhVs)
            UsVs = tf.sigmoid(UsVs)
        
        n_homs = tf.shape(Uh)[0]
        
        # Square error for target scores
        Xt_pred = tf.concat([UhVt, UtVt], axis=0)
        tgt_mask = tf.math.logical_not(tf.is_nan(Xt))
        tgt_err = tf.boolean_mask(Xt - Xt_pred, tgt_mask)
        tgt_sq_err = 0.5 * tf.square(tf.norm(tgt_err))

        # Square error for source scores
        src_mask = tf.math.logical_not(tf.is_nan(Xs))
        src_hom_mask = src_mask[:n_homs]
        src_non_hom_mask = src_mask[n_homs:]
        Xs_homs = Xs[:n_homs]
        Xs_non_homs = Xs[n_homs:]

        # Square error for homolog rows
        src_hom_err = tf.boolean_mask(Xs_homs - UhVs, src_hom_mask)
        src_hom_sq_err = 0.5 * lam_homs * tf.square(tf.norm(src_hom_err))

        # Square error for non homolog rows
        src_non_hom_err = tf.boolean_mask(Xs_non_homs - UsVs, src_non_hom_mask)
        src_non_hom_sq_err = 0.5 * lam_src * tf.square(tf.norm(src_non_hom_err))

        # Add frob-norm regularization on all factors
        reg_U = 0.5 * lam_u * (tf.square(tf.norm(Ut)) + tf.square(tf.norm(Uh)))
        reg_V = 0.5 * lam_v * tf.square(tf.norm(Vt))
        reg_Us = 0.5 * lam_us * tf.square(tf.norm(Us))
        reg_Vs  = 0.5 * lam_vs * tf.square(tf.norm(Vs))
        
        reg = reg_U + reg_V + reg_Us + reg_Vs
        return reg + tgt_sq_err + src_non_hom_sq_err + src_hom_sq_err

    def stop_iterating(self):
        # If number of iterations is has been less than stopping criterion
        # then don't stop
        if len(self.val_r2_hist) < self.early_stop:
            return False
        
        r2s = self.val_r2_hist[-self.early_stop:]

        # continue if any
        min_r2 = r2s[0]
        should_continue = any(r2 > min_r2 for r2 in r2s[1:])
        return not should_continue

    def tgt_r2_score(self, X):
        mask = ~tf.is_nan(X)
        UhVt = tf.matmul(self._UH, self._VT, transpose_b=True)
        UtVt = tf.matmul(self._UT, self._VT, transpose_b=True)
        Xt_pred = tf.concat([UhVt, UtVt], axis=0)

        if self.logistic:
            Xt_pred = tf.sigmoid(Xt_pred)
        
        true_vals = tf.boolean_mask(X, mask)
        imputed_vals = tf.boolean_mask(Xt_pred, mask)

        return self._tf_r2_score_helper(true_vals, imputed_vals)
    
    @staticmethod
    def _tf_r2_score_helper(y_true, y_pred):
        '''
        Compute r2 score using tensorflow variables
        '''
        #assert(y_true.shape == y_pred.shape)
        #assert(len(y_true.shape) == 1)
        total_error = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
        unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))

        r_squared = 1 - tf.div(unexplained_error, total_error)
        return r_squared
    
    def training_curve(self):
        return dict(train=self.train_r2_hist, val=self.val_r2_hist)

    def predict(self, X):
        '''
        Returns array with same shape as X, with NaN values imputed
        '''
        return np.where(np.isnan(X), self.X_fitted, X)
    
    def score(self, X):
        '''        
        Scores imputed values against non NaN values of given matrix under the
        R^2 metric.
        '''
        assert X.shape == self.X_fitted.shape

        mask = ~np.isnan(X)
        imputed_values = self.X_fitted[mask]
        true_values = X[mask]
        # return negative infinity any imputed values are nan
        if np.any(np.isnan(imputed_values)) or np.any(np.isinf(imputed_values)):
            return -np.inf

        r = r2_score(true_values, imputed_values)

        # return negative infinity any r2 is nan
        if np.isnan(r):
            return -np.inf
        else:
            return r
    
    def get_params(self):
        attributes = [
            'lambda_hom',
            'lambda_source',
            'lambda_u',
            'lambda_v',
            'lambda_us',
            'lambda_vs',
            'rank',
            'max_iter',
            'lr',
            'report_every',
            'verbosity',
            'early_stop',
            'early_stop'
        ]
        params = {}
        for attr in attributes:
            params[attr] = getattr(self, attr)
        
        return params

class KXSMF(XSMF):
    def __init__(self, X, X_source, hom_idxs, L_tgt, L_src,
                 X_val=None,
                 lambda_tgt_rl=0.1, # tgt regularized laplacian lambda
                 lambda_src_rl=0.1, # src regularized laplacian lambda
                 lambda_hom=0.1,
                 lambda_source=0.1,
                 lambda_u=0.1,
                 lambda_v=0.1,
                 lambda_us=0.1,
                 lambda_vs=0.1,
                 rank=40,
                 max_iter=150,
                 lr=0.1,
                 report_every=-1,
                 verbosity=logging.INFO,
                 early_stop=10,
                 logistic=False):
        super().__init__(X=X, 
                         X_source=X_source, 
                         hom_idxs=hom_idxs,
                         X_val=X_val,
                         lambda_hom=lambda_hom,
                         lambda_source=lambda_source,
                         lambda_u=lambda_u,
                         lambda_v=lambda_v,
                         lambda_us=lambda_us,
                         lambda_vs=lambda_vs,
                         rank=rank,
                         max_iter=max_iter,
                         lr=lr,
                         report_every=report_every,
                         verbosity=verbosity,
                         early_stop=early_stop,
                         logistic=logistic)
        assert (L_tgt.shape == (X.shape[1], X.shape[1]))
        assert (L_src.shape == (X_source.shape[1], X_source.shape[1]))
        self.L_tgt = L_tgt # target laplacian
        self.L_src = L_src # source laplacian
        self.lambda_tgt_rl = lambda_tgt_rl # target regularized laplacian param
        self.lambda_src_rl = lambda_src_rl # source regularized laplacian param

        # Avoid redundant inverses by only storing \lambda L +  I (inverse of kernel)
        # self.K_inv_tgt = self.L_tgt + self.lambda_tgt_rl * np.eye(len(X))
        # self.K_inv_src = self.L_src + self.lambda_tgt_rl * np.eye(len(X_source))
        self.K_inv_tgt = self.lambda_tgt_rl * self.L_tgt +  np.eye(X.shape[1])
        self.K_inv_src = self.lambda_tgt_rl * self.L_src +  np.eye(X_source.shape[1])
        self._initialize_tf_kernel_vars()

    def _initialize_tf_kernel_vars(self):
        # NOTE. This must be called AFTER the super method which initializes self._{tgt,src}_reordering #TODO this comment no longer applies

        # K_tgt = reorder_square_matrix(self._tgt_reordering, self.K_inv_tgt, mode='to')
        # K_src = reorder_square_matrix(self._src_reordering, self.K_inv_src, mode='to')

        self._K_tgt = tf.constant(self.K_inv_tgt)
        self._K_src = tf.constant(self.K_inv_src)

    def loss(self):
        '''
        Returns tf loss variable
        '''
        return self._loss_helper(Xt=self._XT, 
                                 Xs=self._XS,
                                 Kt=self._K_tgt,
                                 Ks=self._K_src,
                                 Ut=self._UT,
                                 Uh=self._UH,
                                 Us=self._US,
                                 Vt=self._VT,
                                 Vs=self._VS,
                                 lam_homs=self.lambda_hom,
                                 lam_src=self.lambda_source,
                                 lam_u=self.lambda_u, 
                                 lam_v=self.lambda_v,
                                 lam_us=self.lambda_us, 
                                lam_vs=self.lambda_vs,
                                logistic=self.logistic)
    @staticmethod
    def _loss_helper(Xt, Xs, Kt, Ks, Ut, Uh, Us,  Vt, Vs,
             lam_homs, lam_src, lam_u, lam_v, lam_us, lam_vs,
             logistic):

        '''
        Returns
        -------
        TF variables 
        Parameters
        ----------
        Xt : array-like
        Xs : array-like
        Ut : array-like
        Vs : array-like
        homs : List of tuples
        lambda_u : float
        lambda_v : float
        lambda_us : float
        lambda_vs : float
        lambda_homs : float
        lambda_src : float
        '''
        UhVt = tf.matmul(Uh, Vt, transpose_b=True)
        UtVt = tf.matmul(Ut, Vt, transpose_b=True)
        UhVs = tf.matmul(Uh, Vs, transpose_b=True)
        UsVs = tf.matmul(Us, Vs, transpose_b=True)

        if logistic:
            UhVt = tf.sigmoid(UhVt)
            UtVt = tf.sigmoid(UtVt)
            UhVs = tf.sigmoid(UhVs)
            UsVs = tf.sigmoid(UsVs)
        
        n_homs = tf.shape(Uh)[0]
        
        # Square error for target scores
        Xt_pred = tf.concat([UhVt, UtVt], axis=0)
        tgt_mask = tf.math.logical_not(tf.is_nan(Xt))
        tgt_err = tf.boolean_mask(Xt - Xt_pred, tgt_mask)
        tgt_sq_err = 0.5 * tf.square(tf.norm(tgt_err))

        # Square error for source scores
        src_mask = tf.math.logical_not(tf.is_nan(Xs))
        src_hom_mask = src_mask[:n_homs]
        src_non_hom_mask = src_mask[n_homs:]
        Xs_homs = Xs[:n_homs]
        Xs_non_homs = Xs[n_homs:]

        # Square error for homolog rows
        src_hom_err = tf.boolean_mask(Xs_homs - UhVs, src_hom_mask)
        src_hom_sq_err = 0.5 * lam_homs * tf.square(tf.norm(src_hom_err))

        # Square error for non homolog rows
        src_non_hom_err = tf.boolean_mask(Xs_non_homs - UsVs, src_non_hom_mask)
        src_non_hom_sq_err = 0.5 * lam_src * tf.square(tf.norm(src_non_hom_err))

        # Add frob-norm regularization on LHS factors
        reg_U = 0.5 * lam_u * (tf.square(tf.norm(Ut)) + tf.square(tf.norm(Uh)))
        reg_Us = 0.5 * lam_us * tf.square(tf.norm(Us))

        # Add kernel regularization on RHS factors
        reg_V   = 0.5 * lam_v  * tf.trace(tf.transpose(Vt) @ Kt @ Vt)
        reg_Vs  = 0.5 * lam_vs * tf.trace(tf.transpose(Vs) @ Ks @ Vs)
        
        reg = reg_U + reg_V + reg_Us + reg_Vs
        return reg + tgt_sq_err + src_non_hom_sq_err + src_hom_sq_err

### UTILS ###

def selected_idxs_first_ordering(selected_idxs, N):
    ordered = np.arange(N)
    n_selected = len(selected_idxs)
    mask = np.zeros_like(ordered, dtype=bool)
    mask[selected_idxs] = True
    reordering = np.zeros_like(ordered)
    reordering[: n_selected] = selected_idxs
    reordering[n_selected:] = ordered[~mask]
    return reordering

def check_1to1_homs(homs):
    assert np.asarray(homs).shape[1] == 2
    t_genes, s_genes = zip(*homs)
    assert len(t_genes) == len(set(t_genes))
    assert len(s_genes) == len(set(s_genes))

def check_hom_idxs(X_target, X_source, homs):
    # check_square(X_target)
    # check_square(X_source)
    t_genes, s_genes = zip(*homs)
    assert max(t_genes) < len(X_target)
    assert max(s_genes) < len(X_source)

    assert min(t_genes) >= 0
    assert min(s_genes) >= 0

def check_square(X):
    assert len(X.shape) == 2
    assert X.shape[0] == X.shape[1]

# todo: Move this to a utility.py script or something...
def check_gi_obj(obj):
    '''
    Perform sanity check for obj containing GI data
    '''
    mat = obj['m']
    rows = obj['rows']
    cols = obj['cols']

    assert len(rows) == len(cols), \
        'Matrix is not square and has shape {}'.format((len(rows), len(cols)))
    assert len(set(rows) & set(cols)) == len(rows), \
        'Gene names on row and cols are not unique (# names: {}, shape: {}).'.format(
            len(set(rows) & set(cols)), mat.shape)
    assert all(rows == cols) 

def read_homolog_list(fp, to_upper=True):
    ''' Returns tuple of homologs from two column homolgs tsv file '''
    with open(fp, 'r') as f:
        homs = [tuple(l.split()) for l in f]
        if to_upper:
            homs = [(a.upper(), b.upper()) for a, b in homs]
        return homs

def sparsity(X, percent=True):
    '''
    TODO: move to utils
    '''
    s = np.sum(np.isnan(X)) / X.size
    if percent:
        s *= 100.
    return s

def log_gi_stats(log, gi_obj):
    X = gi_obj['m']
    log.info('- Data matrix is %.2f%% missing' % (sparsity(X)))
    log.info('- Data shape: %s', str(X.shape))

def log_hom_stats(log, homs):
    pass

def restrict_homs_to_gis(homs, target_genes, source_genes):
    t_gs = set(target_genes)
    s_gs = set(source_genes)

    return [p for p in homs if p[0] in t_gs and p[1] in s_gs]

def gene2index(genes):
    return dict( (n, i) for i, n in enumerate(genes)) 

def index2gene(genes):
    return dict(enumerate(genes))

def reorder_square_matrix(idxs, X, mode='to'):
    if mode == 'to':
        return X[idxs][:,idxs]
    elif mode == 'from':
        ordering = np.zeros_like(idxs)
        ordering[idxs] = np.arange(len(idxs))
        return reorder_square_matrix(ordering, X, mode='to')
    else:
        raise ValueError

def reorder_matrix_rows(idxs, X, mode='to'):
    if mode == 'to':
        return X[idxs]
    elif mode == 'from':
        ordering = np.zeros_like(idxs)
        ordering[idxs] = np.arange(len(idxs))
        return reorder_matrix_rows(ordering, X, mode='to')
    else:
        raise ValueError

if __name__ == "__main__":
    import argparse
    from matrix_completion import MCScaler
    from sklearn.metrics import r2_score
    np.random.seed(seed=100)
    tf.random.set_random_seed(520)
    # Parser and arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-it', '--target_gis', type=str, required=True)
    parser.add_argument('-is', '--source_gis', type=str, required=True)
    parser.add_argument('-hom', '--homologs', type=str, required=True)
    parser.add_argument('-logs', '--logs', type=str, required=True)
    parser.add_argument('-hf', '--hidden_fraction', type=float, default=0.2)
    args = parser.parse_args()
    setup_logging(args.logs)

    # Load the data
    log = get_logger()
    
    log.info('[Loading target GIs from: %s]', args.target_gis)
    target_gis = np.load(args.target_gis)
    check_gi_obj(target_gis)
    log_gi_stats(log, target_gis)
    
    
    log.info('[Loading source GIs from: %s]', args.source_gis)
    source_gis = np.load(args.source_gis)
    check_gi_obj(source_gis)
    log_gi_stats(log, source_gis)
    
    src_X, tgt_X = source_gis['m'], target_gis['m']
    src_genes = source_gis['rows']
    tgt_genes = target_gis['rows']

    log.info('[Loading homologs from: %s]', args.homologs)
    homologs = read_homolog_list(args.homologs)
    homologs = restrict_homs_to_gis(homologs, 
                                    tgt_genes, src_genes)
    
    check_1to1_homs(homologs)
    log.info('- Found %d homolog pairs in GIs', len(homologs))

    src_g2i = gene2index(src_genes)
    tgt_g2i = gene2index(tgt_genes)

    hom_idxs = np.asarray([(tgt_g2i[t], src_g2i[s]) for t, s in homologs])
    check_hom_idxs(tgt_X, src_X, hom_idxs)

    # Initialize and train the model
    log.info('[Training model - holdout fraction: %0.2f ]', args.hidden_fraction )
    X_train, X_test = mc_train_test_split(tgt_X, frac_hidden=args.hidden_fraction)
    
    use_logistic_model = True
    scaler_mode = 'std' if not use_logistic_model else '0-1'
    print(scaler_mode)
    tgt_scaler = MCScaler(mode=scaler_mode)

    X_train = tgt_scaler.fit_transform(X_train)
    X_test = tgt_scaler.transform(X_test)

    src_X = MCScaler(mode=scaler_mode).fit_transform(src_X)

    xsmf = XSMF(X=X_train,
                X_source= src_X,
                hom_idxs=hom_idxs, 
                lambda_hom=0,
                lambda_source=0,
                lambda_u=.1,
                lambda_v=.1,
                lambda_us=.00,
                lambda_vs=.00,
                max_iter=500,
                lr=0.05,
                rank=35,
                report_every=100,
                logistic=use_logistic_model)
    xsmf.fit()
    train_mask = ~np.isnan(X_train)
    print(r2_score(xsmf.X_fitted[train_mask], X_train[train_mask]))
    src_mask = ~np.isnan(src_X)

    print(r2_score(xsmf.XS_fitted[src_mask], src_X[src_mask]))

    test_mask = ~np.isnan(X_test)
    predicted = (xsmf.X_fitted + xsmf.X_fitted.T)[test_mask] / 2
    true = X_test[test_mask]
    
    print(np.linalg.norm(predicted-true))
    print(r2_score(predicted, true))
    print('#'*10)

    from matrix_completion import PMF
    pmf = PMF(rank=35,
              lambda_f=0.077,
              lambda_h=0.077,
              lr=0.090,
              max_iter=500,
              report_every=100,
              use_sigmoid=True)
    pmf.fit(X_train)
    predicted = pmf.X_fitted[test_mask]
    true = X_test[test_mask]
    
    print(np.linalg.norm(predicted-true))
    print(r2_score(predicted, true))
    
    pass
