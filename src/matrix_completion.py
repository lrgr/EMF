import logging
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from i_o import get_logger
from scipy.special import expit
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error, r2_score

class BaseMatrixCompletion(ABC, BaseEstimator):
    def __init__(self):
        self.X_fitted = None
        self.X = None
        pass

    @abstractmethod
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        '''
        Returns array with same shape as X, with NaN values imputed
        '''

        # This function takes an 'X' unecessarily so sklearn CV methods play nice
        self.check_is_fitted() 
        self.check_same_shape(X, self.X_fitted)

        return np.where(np.isnan(X), self.X_fitted, X)

    def score(self, X, y=None):
        '''        
        Scores imputed values against non NaN values of given matrix under the
        R^2 metric.

        Note: y is ignored and must be None
        '''
        self.check_is_none(y)
        self.check_is_fitted()
        self.check_same_shape(X, self.X_fitted)

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
    
    # Useful Checks

    def check_is_fitted(self):
        if self.X_fitted is None or self.X is None:
            raise NotFittedError('Matrix completion has not been fitted') 

    
    @staticmethod
    def check_same_shape(X1, X2):
        if X1.shape != X2.shape:
            raise ValueError('Shapes {} and {} are not identical'.format(
                              X1.shape, X2.shape))

    @staticmethod
    def check_equal(X1, X2):
        if not np.allclose(X1, X2, equal_nan=True):
            raise ValueError('Arrays are not equal')
    
    @staticmethod
    def check_no_shared_non_nan(X1, X2):
        BaseMatrixCompletion.check_same_shape(X1, X2)

        shared = ~np.isnan(X1) & ~np.isnan(X2)
        if np.any(shared):
            raise ValueError('Arrays share non-nan entries')
    
    @staticmethod
    def check_is_none(x):
        if x is not None:
            raise ValueError('Argument must not be supplied or must be None')
        

class MeanMC(BaseMatrixCompletion):
    def fit(self, X, y=None):
        super().__init__()
        self.X = np.copy(X)
        mean = np.mean(X[~np.isnan(X)])
        self.X_fitted = np.copy(X)
        self.X_fitted[np.isnan(X)] = mean
        return self

class FillMC(BaseMatrixCompletion):
    def __init__(self, fillna=0.):
        super().__init__()
        self.fillna = 0.

    def fit(self, X, y=None):
        self.X = np.copy(X)
        self.X_fitted = np.copy(X)
        self.X_fitted[np.isnan(X)] = self.fillna
        return self
        
class MF_GradientDescentMixin(ABC):
    def gd_fit(self):
        self._F, self._H = tf.Variable(self.F), tf.Variable(self.H)

        train_loss = self.loss(self.X)
        train_r2_tf = self.tf_r2_score(self.X)

        if self.use_validation():
            val_r2_tf = self.tf_r2_score(self.X_val)

        #optim = tf.train.GradientDescentOptimizer(self.lr)
        optim = tf.train.AdamOptimizer(self.lr)
        train_step = optim.minimize(train_loss)
        init = tf.global_variables_initializer()
            
        _early_stopped = False
        with tf.Session() as sess:
            sess.run(init)
            for i in range(self.max_iter):
                sess.run(train_step)
                train_err, train_r2 = sess.run([train_loss, train_r2_tf])
                self.train_loss_hist.append(train_err)
                self.train_r2_hist.append(train_r2)
                if self.use_validation():
                    val_r2 = sess.run(val_r2_tf)
                    self.val_r2_hist.append(val_r2)

                if self.report_every > 1 and i % self.report_every == 0:
                    self.log.info("\titer={:5d} | Training   | Cost={:8.4f}".format(i, train_err))
                    self.log.info("\titer={:5d} | Training   | R2={:8.4f}".format(i, train_r2))
                    if self.use_validation():
                        self.log.info("\titer={:5d} | Validation | R2={:8.4f}".format(i, val_r2))

                if np.isnan(train_err):
                    self.log.warn('\tquitting training early due to NaN loss')
                    break

                if self.use_validation() and self.early_stop > 0:
                    if self.stop_iterating():
                        self.log.info("\tstopping early due to non-decreasing validation loss")
                        _early_stopped = True
                        break

            if not _early_stopped:
                self.log.info('\tstopping due to max-iterations reached')
            
            # Report last iteration losses
            if self.report_every > 1:
                self.log.info("\titer={:5d} | Training   | Cost={:8.4f}".format(i, train_err))
                if self.use_validation():
                        self.log.info("\titer={:5d} | Training   | R2={:8.4f}".format(i, train_r2))
                        self.log.info("\titer={:5d} | Validation | R2={:8.4f}".format(i, val_r2))

            learnt_F = sess.run(self._F)
            learnt_H = sess.run(self._H)

        # Store how many iterations we ran for for convenience
        self.iters_ran = len(self.train_loss_hist)

        return learnt_F, learnt_H
    
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
    
    def use_validation(self):
        return hasattr(self, 'X_val')
    
    def tf_r2_score(self, X):
        _mask = ~np.isnan(X)
        _mask = tf.constant(_mask)
        _X = tf.constant(X)

        FH = tf.matmul(tf.transpose(self._F), self._H)
        if self.use_sigmoid:
            FH = tf.sigmoid(0.5 * FH)
        
        true_vals = tf.boolean_mask(_X, _mask)
        imputed_vals = tf.boolean_mask(FH, _mask)
        return tf_r2_score(true_vals, imputed_vals)
    
    def training_curve(self):
        return dict(train=self.train_r2_hist, val=self.val_r2_hist)

        
class KPMF(BaseMatrixCompletion, MF_GradientDescentMixin):
    def __init__(self, 
                 lambda_f=1,
                 lambda_h=1,
                 rank=40, 
                 max_iter=150, 
                 lr=0.1,
                 early_stop=10,
                 use_sigmoid=True,
                 report_every=-1,
                 verbosity=logging.INFO):

        super().__init__()
        self.log = get_logger(verbosity)

        # Model Hyperparameters
        self.lambda_f = lambda_f
        self.lambda_h = lambda_h
        self.rank = rank
        self.max_iter = max_iter
        self.lr = lr
        self.early_stop = early_stop
        self.use_sigmoid = use_sigmoid

        # Logging settings
        self.report_every = report_every
        self.verbosity = verbosity
        
        # History
        self.iters_ran = -1
        self.train_loss_hist = []
        self.train_r2_hist = []
        self.val_r2_hist = None
    
    def fit(self, X, y=None, X_val=None, F_kernel=None, H_kernel=None):
        '''
        X := (train_matrix, F_kernel, K_kernel)
        '''
        assert y is None
        if F_kernel is None: 
            F_kernel = np.eye(X.shape[0])
        if H_kernel is None:
            H_kernel = np.eye(X.shape[1])
            
        assert ((X.shape[0], X.shape[0]) == F_kernel.shape)
        assert ((X.shape[1], X.shape[1]) == H_kernel.shape)

        if X_val is not None:
            self.check_same_shape(X_val, X)
            self.check_no_shared_non_nan(X_val, X)
            self.X_val = np.copy(X_val)
            self.val_r2_hist = []

        self.X = np.copy(X)
        self.train_mask = ~np.isnan(self.X)
        self.M, self.N = self.X.shape

        assert self.M > self.rank
        assert self.N > self.rank

        self.F_kernel = np.copy(F_kernel)
        self.H_kernel = np.copy(H_kernel)

        temp_F = np.random.randn(self.rank, self.M).astype(np.float64)
        self.F = np.divide(temp_F, temp_F.max())

        temp_H = np.random.randn(self.rank, self.N).astype(np.float64)
        self.H = np.divide(temp_H, temp_H.max())

        self.S_F = np.linalg.inv(self.F_kernel)
        self.S_H = np.linalg.inv(self.H_kernel)

        self.F, self.H = self.gd_fit()
        self.X_fitted = np.dot(self.F.T, self.H)

        if self.use_sigmoid:
            self.X_fitted = expit(0.5 * self.X_fitted)
        tf.reset_default_graph()
        return self

    def loss(self, X):
        _mask = ~np.isnan(X)
        _mask = tf.constant(_mask)
        _X = tf.constant(X)

        S_F = tf.constant(self.S_F, dtype=tf.float64)
        S_H = tf.constant(self.S_H, dtype=tf.float64)

        FH = tf.matmul(tf.transpose(self._F), self._H)

        if self.use_sigmoid:
            FH = tf.sigmoid(0.5 * FH)
        
        non_reg_loss = reduce_sum_det(tf.pow(tf.boolean_mask(_X, _mask) 
                                    - tf.boolean_mask(FH, _mask), 2))

        lambda_f = tf.constant(0.5 * self.lambda_f, dtype=tf.float64)
        lambda_h = tf.constant(0.5 * self.lambda_h, dtype=tf.float64)

        reg_F = tf.trace(self._F @ S_F @ tf.transpose(self._F))
        reg_H = tf.trace(self._H @ S_H @ tf.transpose(self._H))

        return  0.5 * non_reg_loss \
                + 0.5 * lambda_f * reg_F \
                + 0.5 * lambda_h * reg_H

class KPMF_b(BaseMatrixCompletion):
    def __init__(self, 
                 rank=40, 
                 lambda_f=0.01, 
                 lambda_h=None,
                 lambda_b=0.01,
                 use_sigmoid=False,
                 max_iter=150, 
                 lr=0.1,
                 report_every=-1,
                 verbosity=logging.INFO,
                 early_stop=10):
        super().__init__()
        self.log = get_logger(verbosity)

        # Model hyperparameters
        self.rank = rank
        # set regularization weights to be the same if lambda_h is None
        self.lambda_f = lambda_f
        self.lambda_h = lambda_f if lambda_h is None else lambda_h
        self.lambda_b = lambda_b
        self.use_sigmoid = use_sigmoid
        self.max_iter = max_iter
        self.lr = lr
        self.early_stop = early_stop

        # Logging settings
        self.report_every = report_every
        self.verbosity = verbosity

        self.iters_ran = -1
        self.train_loss_hist = []
        self.train_r2_hist = []
        self.val_r2_hist = None

        
    def fit(self, X, y=None, X_val=None, F_kernel=None, H_kernel=None):
        '''
        X := (train_matrix, F_kernel, K_kernel)
        '''
        assert y is None
        if F_kernel is None: 
            F_kernel = np.eye(X.shape[0])
        if H_kernel is None:
            H_kernel = np.eye(X.shape[1])
            
        assert ((X.shape[0], X.shape[0]) == F_kernel.shape)
        assert ((X.shape[1], X.shape[1]) == H_kernel.shape)

        if X_val is not None:
            self.check_same_shape(X_val, X)
            self.check_no_shared_non_nan(X_val, X)
            self.X_val = np.copy(X_val)
            self.val_r2_hist = []

        self.X = np.copy(X)

        self.train_mask = ~np.isnan(self.X)
        self.M, self.N = self.X.shape

        assert self.M > self.rank
        assert self.N > self.rank

        self.F_kernel = np.copy(F_kernel)
        self.H_kernel = np.copy(H_kernel)

        self.S_F = np.linalg.inv(self.F_kernel)
        self.S_H = np.linalg.inv(self.H_kernel)

        self.bias_row = np.nansum(X, axis=1, keepdims=True) / 2
        self.bias_row = self.bias_row / np.nansum(np.isnan(X), axis=1, keepdims=True)
        self.bias_col = np.nansum(X, axis=0, keepdims=True) / 2
        self.bias_col = self.bias_col / np.nansum(np.isnan(X), axis=0, keepdims=True)

        temp_F = np.random.randn(self.rank, self.M).astype(np.float64)
        self.F = np.divide(temp_F, temp_F.max())

        temp_H = np.random.randn(self.rank, self.N).astype(np.float64)
        self.H = np.divide(temp_H, temp_H.max())

        self.F, self.H, self.bias_row, self.bias_col = self.gd_fit()
        self.X_fitted = np.dot(self.F.T, self.H) + self.bias_row + self.bias_col
        
        if self.use_sigmoid:
            self.X_fitted = expit(0.5 * self.X_fitted)
        tf.reset_default_graph()
        return self
    
    def loss(self, X):
        '''
        return loss w.r.t masked values
        '''
        _mask = ~np.isnan(X)
        _mask = tf.constant(_mask)
        _X = tf.constant(X)

        lambda_f = tf.constant(0.5 * self.lambda_f, dtype=tf.float64)
        lambda_h = tf.constant(0.5 * self.lambda_h, dtype=tf.float64)
        lambda_b = tf.constant(0.5 * self.lambda_b, dtype=tf.float64)

        FH = tf.matmul(tf.transpose(self._F), self._H) + self._br + self._bc

        S_F = tf.constant(self.S_F, dtype=tf.float64)
        S_H = tf.constant(self.S_H, dtype=tf.float64)

        if self.use_sigmoid:
            FH = tf.sigmoid(0.5 * FH)

        non_reg_loss = 0.5 * tf.reduce_sum(tf.pow(tf.boolean_mask(_X, _mask) 
                                   - tf.boolean_mask(FH, _mask), 2))

        lambda_f = tf.constant(0.5 * self.lambda_f, dtype=tf.float64)
        lambda_h = tf.constant(0.5 * self.lambda_h, dtype=tf.float64)
        reg_F = lambda_f * tf.trace(self._F @ S_F @ tf.transpose(self._F))
        reg_H = lambda_h * tf.trace(self._H @ S_H @ tf.transpose(self._H))
        
        reg_b = reduce_sum_det(self._br * self._br) + reduce_sum_det(self._bc * self._bc)

        return non_reg_loss \
            + lambda_f * reg_F + lambda_h * reg_H \
            + lambda_b * reg_b
            #+ tf.multiply(lambda_h, tf.reduce_sum(tf.multiply(self._F, self._F))) \
            #+ tf.multiply(lambda_h, tf.reduce_sum(tf.multiply(self._H, self._H)))

    def gd_fit(self):
        self._F, self._H = tf.Variable(self.F), tf.Variable(self.H)
        self._br, self._bc = tf.Variable(self.bias_row), tf.Variable(self.bias_col)

        train_loss = self.loss(self.X)
        train_r2_tf = self.tf_r2_score(self.X)

        if self.use_validation():
            val_r2_tf = self.tf_r2_score(self.X_val)

        #optim = tf.train.GradientDescentOptimizer(self.lr)
        optim = tf.train.AdamOptimizer(self.lr)
        train_step = optim.minimize(train_loss)
        init = tf.global_variables_initializer()
            
        _early_stopped = False
        with tf.Session() as sess:
            sess.run(init)
            for i in range(self.max_iter):
                sess.run(train_step)
                train_err, train_r2 = sess.run([train_loss, train_r2_tf])
                self.train_loss_hist.append(train_err)
                self.train_r2_hist.append(train_r2)
                if self.use_validation():
                    val_r2 = sess.run(val_r2_tf)
                    self.val_r2_hist.append(val_r2)

                if self.report_every > 1 and i % self.report_every == 0:
                    self.log.info("\titer={:5d} | Training   | Cost={:8.4f}".format(i, train_err))
                    self.log.info("\titer={:5d} | Training   | R2={:8.4f}".format(i, train_r2))
                    if self.use_validation():
                        self.log.info("\titer={:5d} | Validation | R2={:8.4f}".format(i, val_r2))

                if np.isnan(train_err):
                    self.log.warn('\tquitting training early due to NaN loss')
                    break

                if self.use_validation() and self.early_stop > 0:
                    if self.stop_iterating():
                        self.log.info("\tstopping early due to non-decreasing validation loss")
                        _early_stopped = True
                        break

            if not _early_stopped:
                self.log.info('\tstopping due to max-iterations reached')
            
            # Report last iteration losses
            if self.report_every > 1:
                self.log.info("\titer={:5d} | Training   | Cost={:8.4f}".format(i, train_err))
                if self.use_validation():
                        self.log.info("\titer={:5d} | Training   | R2={:8.4f}".format(i, train_r2))
                        self.log.info("\titer={:5d} | Validation | R2={:8.4f}".format(i, val_r2))

            learnt_F = sess.run(self._F)
            learnt_H = sess.run(self._H)
            learnt_br = sess.run(self._br)
            learnt_bc = sess.run(self._bc)

        # Store how many iterations we ran for for convenience
        self.iters_ran = len(self.train_loss_hist)

        return learnt_F, learnt_H, learnt_br, learnt_bc
    
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
    
    def use_validation(self):
        return hasattr(self, 'X_val')
    
    def tf_r2_score(self, X):
        _mask = ~np.isnan(X)
        _mask = tf.constant(_mask)
        _X = tf.constant(X)

        FH = tf.matmul(tf.transpose(self._F), self._H) + self._br + self._bc
        if self.use_sigmoid:
            FH = tf.sigmoid(0.5 * FH)
        
        true_vals = tf.boolean_mask(_X, _mask)
        imputed_vals = tf.boolean_mask(FH, _mask)
        return tf_r2_score(true_vals, imputed_vals)
    
    def training_curve(self):
        return dict(train=self.train_r2_hist, val=self.val_r2_hist)

class PMF_b(BaseMatrixCompletion):
    '''
    PMF with bias
    x = u'v + b_i + c_j
    '''
    def __init__(self, 
                 rank=40, 
                 lambda_f=0.01, 
                 lambda_h=None,
                 lambda_b=0.01,
                 use_sigmoid=False,
                 max_iter=150, 
                 lr=0.1,
                 report_every=-1,
                 verbosity=logging.INFO,
                 early_stop=10):
        super().__init__()
        self.log = get_logger(verbosity)

        # Model hyperparameters
        self.rank = rank
        # set regularization weights to be the same if lambda_h is None
        self.lambda_f = lambda_f
        self.lambda_h = lambda_f if lambda_h is None else lambda_h
        self.lambda_b = lambda_b
        self.use_sigmoid = use_sigmoid
        self.max_iter = max_iter
        self.lr = lr
        self.early_stop = early_stop

        # Logging settings
        self.report_every = report_every
        self.verbosity = verbosity

        self.iters_ran = -1
        self.train_loss_hist = []
        self.train_r2_hist = []
        self.val_r2_hist = None

        
    def fit(self, X, y=None, X_val=None):
        assert y is None
        self.train_mask = ~np.isnan(X)

        self.M, self.N = X.shape

        assert self.M > self.rank
        assert self.N > self.rank

        self.X = np.copy(X)

        self.bias_row = np.nansum(X, axis=1, keepdims=True) / 2
        self.bias_row = self.bias_row / np.nansum(np.isnan(X), axis=1, keepdims=True)
        self.bias_col = np.nansum(X, axis=0, keepdims=True) / 2
        self.bias_col = self.bias_col / np.nansum(np.isnan(X), axis=0, keepdims=True)

        if X_val is not None:
            self.check_same_shape(X_val, X)
            self.check_no_shared_non_nan(X_val, X)
            self.X_val = np.copy(X_val)
            self.val_r2_hist = []

        temp_F = np.random.randn(self.rank, self.M).astype(np.float64)
        self.F = np.divide(temp_F, temp_F.max())

        temp_H = np.random.randn(self.rank, self.N).astype(np.float64)
        self.H = np.divide(temp_H, temp_H.max())

        self.F, self.H, self.bias_row, self.bias_col = self.gd_fit()
        self.X_fitted = np.dot(self.F.T, self.H) + self.bias_row + self.bias_col
        
        if self.use_sigmoid:
            self.X_fitted = expit(0.5 * self.X_fitted)
        tf.reset_default_graph()
        return self
    
    def loss(self, X):
        '''
        return loss w.r.t masked values
        '''
        _mask = ~np.isnan(X)
        _mask = tf.constant(_mask)
        _X = tf.constant(X)

        lambda_f = tf.constant(0.5 * self.lambda_f, dtype=tf.float64)
        lambda_h = tf.constant(0.5 * self.lambda_h, dtype=tf.float64)
        lambda_b = tf.constant(0.5 * self.lambda_b, dtype=tf.float64)

        FH = tf.matmul(tf.transpose(self._F), self._H) + self._br + self._bc

        if self.use_sigmoid:
            FH = tf.sigmoid(0.5 * FH)

        non_reg_loss = 0.5 * tf.reduce_sum(tf.pow(tf.boolean_mask(_X, _mask) 
                                   - tf.boolean_mask(FH, _mask), 2))

        reg_F = reduce_sum_det(self._F * self._F)
        reg_H = reduce_sum_det(self._H * self._H)
        reg_b = reduce_sum_det(self._br * self._br) + reduce_sum_det(self._bc * self._bc)

        return non_reg_loss \
            + lambda_f * reg_F + lambda_h * reg_H \
            + lambda_b * reg_b
            #+ tf.multiply(lambda_h, tf.reduce_sum(tf.multiply(self._F, self._F))) \
            #+ tf.multiply(lambda_h, tf.reduce_sum(tf.multiply(self._H, self._H)))

    def gd_fit(self):
        self._F, self._H = tf.Variable(self.F), tf.Variable(self.H)
        self._br, self._bc = tf.Variable(self.bias_row), tf.Variable(self.bias_col)

        train_loss = self.loss(self.X)
        train_r2_tf = self.tf_r2_score(self.X)

        if self.use_validation():
            val_r2_tf = self.tf_r2_score(self.X_val)

        #optim = tf.train.GradientDescentOptimizer(self.lr)
        optim = tf.train.AdamOptimizer(self.lr)
        train_step = optim.minimize(train_loss)
        init = tf.global_variables_initializer()
            
        _early_stopped = False
        with tf.Session() as sess:
            sess.run(init)
            for i in range(self.max_iter):
                sess.run(train_step)
                train_err, train_r2 = sess.run([train_loss, train_r2_tf])
                self.train_loss_hist.append(train_err)
                self.train_r2_hist.append(train_r2)
                if self.use_validation():
                    val_r2 = sess.run(val_r2_tf)
                    self.val_r2_hist.append(val_r2)

                if self.report_every > 1 and i % self.report_every == 0:
                    self.log.info("\titer={:5d} | Training   | Cost={:8.4f}".format(i, train_err))
                    self.log.info("\titer={:5d} | Training   | R2={:8.4f}".format(i, train_r2))
                    if self.use_validation():
                        self.log.info("\titer={:5d} | Validation | R2={:8.4f}".format(i, val_r2))

                if np.isnan(train_err):
                    self.log.warn('\tquitting training early due to NaN loss')
                    break

                if self.use_validation() and self.early_stop > 0:
                    if self.stop_iterating():
                        self.log.info("\tstopping early due to non-decreasing validation loss")
                        _early_stopped = True
                        break

            if not _early_stopped:
                self.log.info('\tstopping due to max-iterations reached')
            
            # Report last iteration losses
            if self.report_every > 1:
                self.log.info("\titer={:5d} | Training   | Cost={:8.4f}".format(i, train_err))
                if self.use_validation():
                        self.log.info("\titer={:5d} | Training   | R2={:8.4f}".format(i, train_r2))
                        self.log.info("\titer={:5d} | Validation | R2={:8.4f}".format(i, val_r2))

            learnt_F = sess.run(self._F)
            learnt_H = sess.run(self._H)
            learnt_br = sess.run(self._br)
            learnt_bc = sess.run(self._bc)

        # Store how many iterations we ran for for convenience
        self.iters_ran = len(self.train_loss_hist)

        return learnt_F, learnt_H, learnt_br, learnt_bc
    
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
    
    def use_validation(self):
        return hasattr(self, 'X_val')
    
    def tf_r2_score(self, X):
        _mask = ~np.isnan(X)
        _mask = tf.constant(_mask)
        _X = tf.constant(X)

        FH = tf.matmul(tf.transpose(self._F), self._H) + self._br + self._bc
        if self.use_sigmoid:
            FH = tf.sigmoid(0.5 * FH)
        
        true_vals = tf.boolean_mask(_X, _mask)
        imputed_vals = tf.boolean_mask(FH, _mask)
        return tf_r2_score(true_vals, imputed_vals)
    
    def training_curve(self):
        return dict(train=self.train_r2_hist, val=self.val_r2_hist)

class PMF(BaseMatrixCompletion, MF_GradientDescentMixin):
    def __init__(self, 
                 rank=40, 
                 lambda_f=0.01, 
                 lambda_h=None,
                 use_sigmoid=True,
                 max_iter=150, 
                 lr=0.1,
                 report_every=-1,
                 verbosity=logging.INFO,
                 early_stop=10):
        super().__init__()
        self.log = get_logger(verbosity)

        # Model hyperparameters
        self.rank = rank
        # set regularization weights to be the same if lambda_h is None
        self.lambda_f = lambda_f
        self.lambda_h = lambda_f if lambda_h is None else lambda_h
        self.use_sigmoid = use_sigmoid
        self.max_iter = max_iter
        self.lr = lr
        self.early_stop = early_stop

        # Logging settings
        self.report_every = report_every
        self.verbosity = verbosity

        self.iters_ran = -1
        self.train_loss_hist = []
        self.train_r2_hist = []
        self.val_r2_hist = None

    
    def fit(self, X, y=None, X_val=None):
        assert y is None
        self.train_mask = ~np.isnan(X)

        self.M, self.N = X.shape

        assert self.M > self.rank
        assert self.N > self.rank

        self.X = np.copy(X)

        if X_val is not None:
            self.check_same_shape(X_val, X)
            self.check_no_shared_non_nan(X_val, X)
            self.X_val = np.copy(X_val)
            self.val_r2_hist = []

        temp_F = np.random.randn(self.rank, self.M).astype(np.float64)
        self.F = np.divide(temp_F, temp_F.max())

        temp_H = np.random.randn(self.rank, self.N).astype(np.float64)
        self.H = np.divide(temp_H, temp_H.max())

        self.F, self.H = self.gd_fit()
        self.X_fitted = np.dot(self.F.T, self.H)
        
        if self.use_sigmoid:
            self.X_fitted = expit(0.5 * self.X_fitted)
        tf.reset_default_graph()
        return self
    
    def loss(self, X):
        '''
        return loss w.r.t masked values
        '''
        _mask = ~np.isnan(X)
        _mask = tf.constant(_mask)
        _X = tf.constant(X)

        lambda_f = tf.constant(0.5 * self.lambda_f, dtype=tf.float64)
        lambda_h = tf.constant(0.5 * self.lambda_h, dtype=tf.float64)

        FH = tf.matmul(tf.transpose(self._F), self._H)
        if self.use_sigmoid:
            FH = tf.sigmoid(0.5 * FH)

        non_reg_loss = 0.5 * tf.reduce_sum(tf.pow(tf.boolean_mask(_X, _mask) 
                                   - tf.boolean_mask(FH, _mask), 2))

        reg_F = reduce_sum_det(self._F * self._F)
        reg_H = reduce_sum_det(self._H * self._H)

        return non_reg_loss \
            + lambda_f * reg_F + lambda_h * reg_H
            #+ tf.multiply(lambda_h, tf.reduce_sum(tf.multiply(self._F, self._F))) \
            #+ tf.multiply(lambda_h, tf.reduce_sum(tf.multiply(self._H, self._H))) 

class MCScaler(TransformerMixin):

    def __init__(self, mode='std'):
        _modes = ['std', '0-1']
        assert (mode in _modes)

        self.mode = mode
    
    def fit(self, X, y=None):
        self.min = np.nanmin(X)
        self.max = np.nanmax(X)
        self.mean = np.nanmean(X)
        self.std = np.nanstd(X)
        return self
    
    def transform(self, X):
        if self.mode == '0-1':
            _X = (X - self.min) / (self.max - self.min)
        else:
            _X = (X - self.mean) / self.std
        return _X

    def inverse_transform(self, X):
        if self.mode == '0-1':
            _X = X * (self.max - self.min) + self.min
        else:
            _X = X * self.std + self.mean
        return _X

### Tensorflow utilities
def reduce_sum_det(x):
    v = tf.reshape(x, [1, -1])
    return tf.reshape(tf.matmul(v, tf.ones_like(v), transpose_b=True), [])

def tf_r2_score(y_true, y_pred):
    '''
    Compute r2 score using tensorflow variables
    '''
    #assert(y_true.shape == y_pred.shape)
    #assert(len(y_true.shape) == 1)
    total_error = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))

    r_squared = 1 - tf.div(unexplained_error, total_error)
    return r_squared

### Other useful functions
# def nrmse(X_true, X_predicted):
#     rmse = np.sqrt(mean_squared_error(X_true, X_predicted))
#     return rmse / np.std(X_true)

def mc_train_test_split(X, frac_hidden, check_sym=True):
    N, M = X.shape

    measured_upper_idx = np.array([ (i,j) for i, j in zip(*np.where(~np.isnan(X)))
                                    if i >= j ])
    
    n_known = len(measured_upper_idx)
    shuffle = np.random.permutation(n_known)
    n_hidden = int(frac_hidden * n_known)
    
    train_upper_idx = measured_upper_idx[shuffle[n_hidden:]]
    train_idx = [ (i, j) for i, j in train_upper_idx ] + \
                [ (j, i) for i, j in train_upper_idx ]

    val_upper_idx = measured_upper_idx[shuffle[:n_hidden]]
    val_idx = [ (i, j) for i, j in val_upper_idx ] + \
              [ (j, i) for i, j in val_upper_idx ]

    I_train = np.zeros((M, N), dtype=bool)
    I_train[tuple(np.array(train_idx).T)] = True
    I_val = np.zeros((M, N), dtype=bool)
    I_val[tuple(np.array(val_idx).T)] = True

    X_train = np.copy(X)
    X_test =  np.copy(X)

    X_train[~I_train] = np.nan
    X_test[~I_val] = np.nan
    assert np.allclose(I_val, I_val.T)
    assert np.allclose(I_train, I_train.T)
    
    if check_sym:
        assert np.allclose(X, X.T, equal_nan=True)
        assert np.allclose(X_train, X_train.T, equal_nan=True)
        assert np.allclose(X_test, X_test.T, equal_nan = True)

    return X_train, X_test
