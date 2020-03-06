import argparse
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from i_o import get_logger, log_dict, setup_logging
from sklearn.metrics import r2_score

class XSMF(object):
    def __init__(self, X_tgt, X_src, 
                 sim_scores,
                 X_val=None,
                 lambda_u=0.1,
                 lambda_v=0.1,
                 lambda_us=0.1,
                 lambda_vs=0.1,
                 lambda_sim=0.1,
                 lambda_src=0.1,
                 rank=40,
                 max_iter=150,
                 lr=0.1,
                 report_every=-1,
                 verbosity=logging.INFO,
                 early_stop=10):
        
        self.log = get_logger(verbosity)
        self.X = X_tgt
        self.X_source = X_src
        self.X_val = X_val
        self.sim_scores = sim_scores
        self.verbosity=verbosity

        self.rank = int(rank)
        self.lr = lr
        self.max_iter = max_iter
        self.report_every = report_every
        self.early_stop = early_stop

        self.lambda_u = lambda_u # reg param for target U
        self.lambda_v = lambda_v # reg param for target V
        self.lambda_us = lambda_us # reg param for source U
        self.lambda_vs = lambda_vs # reg param for target V

        self.lambda_sim = lambda_sim # reg param for similarity
        self.lambda_src = lambda_src
        
        # Initialize source and target latent factors
        self._initialize_factors()

        self.src_mask = ~np.isnan(self.X_source)
        self.tgt_mask = ~np.isnan(self.X)

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

        if self.X_val is not None:
            self._X_val  = tf.constant(self.X_val)

        self._XT  = tf.constant(self.X)
        self._XS = tf.constant(self.X_source)

        init = tf.initializers.random_normal()
        self._UT = tf.get_variable('UT', shape=(self.N, self.rank),
                                    initializer=init, dtype=tf.float64)
        self._US = tf.get_variable('US', shape=(self.NS, self.rank),
                                    initializer=init, dtype=tf.float64)
        self._VT = tf.get_variable('VT', shape=(self.M, self.rank),
                                    initializer=init, dtype=tf.float64)
        self._VS = tf.get_variable('VS', shape=(self.MS, self.rank),
                                    initializer=init, dtype=tf.float64)
    
    def loss(self):
        XT_pred = tf.matmul(self._UT, self._VT, transpose_b=True)
        XS_pred = tf.matmul(self._US, self._VS, transpose_b=True)
        
        tgt_mask = tf.math.logical_not(tf.is_nan(self._XT))
        tgt_err = tf.boolean_mask(self._XT - XT_pred, tgt_mask)
        tgt_sq_err = 0.5 * tf.square(tf.norm(tgt_err))

        src_mask = tf.math.logical_not(tf.is_nan(self._XS))
        src_err = tf.boolean_mask(self._XS - XS_pred, src_mask)
        src_sq_err = 0.5 * self.lambda_src * tf.square(tf.norm(src_err))


        reg_Vs = 0.5 * self.lambda_vs * tf.square(tf.norm(self._VS))
        reg_Vt  = 0.5 * self.lambda_v * tf.square(tf.norm(self._VT))
        reg_Us = 0.5 * self.lambda_us * tf.square(tf.norm(self._US))
        reg_Ut  = 0.5 * self.lambda_u * tf.square(tf.norm(self._UT))

        # TODO:
        VtSVs = tf.matmul(tf.matmul(self._VT, self.sim_scores, transpose_a=True), self._VS)
        sim_sqr_err =  -tf.trace(VtSVs)
        sim_sqr_err_reg = 0.5 * self.lambda_sim * sim_sqr_err

        reg = reg_Vs + reg_Vt + reg_Us + reg_Ut
        return src_sq_err + tgt_sq_err + reg + sim_sqr_err_reg, \
               sim_sqr_err_reg, \
               tgt_sq_err, \
               src_sq_err, \
               reg
                
    
    def fit(self):        
        loss, sim_sqr_diff, tgt_sq_err, src_sq_err, reg = self.loss()
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
                err, train_r2, err_sims, err_tgt, err_src, err_reg = sess.run([loss, train_r2_tf, sim_sqr_diff,  tgt_sq_err, src_sq_err, reg])
                self.train_loss_hist.append(err)
                self.train_r2_hist.append(train_r2)

                if self.X_val is not None:
                    val_r2 = sess.run(val_r2_tf)
                    self.val_r2_hist.append(val_r2)
                sess.run(train_step)

                # Reporting
                if self.report_every >= 1 and i % self.report_every == 0:
                    self.log.info("\titer={:5d} | Training   | Cost={:8.4f}".format(i, err))
                    self.log.info("\titer={:5d} | Training   | Sim Sqr Diff={:8.4f}".format(i, err_sims))
                    self.log.info("\titer={:5d} | Training   | tgt Sqr Diff={:8.4f}".format(i, err_tgt))
                    self.log.info("\titer={:5d} | Training   | src Sqr Diff={:8.4f}".format(i, err_src))
                    self.log.info("\titer={:5d} | Training   | reg Sqr Diff={:8.4f}".format(i, err_reg))

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
            Ut, Us, Vt, Vs = \
                sess.run([self._UT, self._US, self._VT, self._VS])
            
        self.XS_fitted = np.dot(Us, Vs.T)
        self.X_fitted = np.dot(Ut, Vt.T)


        # TODO: self.U, self.US, self.V, self.VS need to be reordered
        
        self.U = Ut
        self.V = Vt
        self.US = Us
        self.VS = Vs

        assert np.allclose(self.X_fitted, np.dot(self.U, self.V.T))
        assert np.allclose(self.XS_fitted, np.dot(self.US, self.VS.T))

        # Store how many iterations we ran for for convenience
        self.iters_ran = len(self.train_loss_hist)

        return self

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
        XT_pred = tf.matmul(self._UT, self._VT, transpose_b=True)
        
        true_vals = tf.boolean_mask(X, mask)
        imputed_vals = tf.boolean_mask(XT_pred, mask)

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
            'lambda_sim',
            'lambda_src',
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
        ]
        params = {}
        for attr in attributes:
            params[attr] = getattr(self, attr)
        
        return params

class KXSMF(XSMF):
    def __init__(self, X_tgt, X_src,
                 sim_scores, L_tgt, L_src,
                 X_val=None,
                 lambda_tgt_rl=0.1, # tgt regularized laplacian lambda
                 lambda_src_rl=0.1, # src regularized laplacian lambda
                 lambda_u=0.1,
                 lambda_v=0.1,
                 lambda_us=0.1,
                 lambda_vs=0.1,
                 lambda_sim=0.1,
                 lambda_src=0.1,
                 rank=40,
                 max_iter=150,
                 lr=0.1,
                 report_every=-1,
                 verbosity=logging.INFO,
                 early_stop=10):
        super().__init__(
            X_tgt=X_tgt,
            X_src=X_src,
            sim_scores=sim_scores,
            X_val=X_val,
            lambda_sim=lambda_sim,
            lambda_src=lambda_src,
            lambda_u=lambda_u,
            lambda_v=lambda_v,
            lambda_us=lambda_us,
            lambda_vs=lambda_vs,
            rank=rank,
            max_iter=max_iter,
            lr=lr,
            report_every=report_every,
            verbosity=verbosity,
            early_stop=early_stop
        )

        self.L_tgt = L_tgt # target laplacian
        self.L_src = L_src # source laplacian
        self.lambda_tgt_rl = lambda_tgt_rl # target regularized laplacian param
        self.lambda_src_rl = lambda_src_rl # source regularized laplacian param

        self.K_inv_tgt = self.lambda_tgt_rl * self.L_tgt +  np.eye(X_tgt.shape[0])
        self.K_inv_src = self.lambda_tgt_rl * self.L_src +  np.eye(X_src.shape[0])

        self._initialize_tf_kernel_vars()

    def _initialize_tf_kernel_vars(self):
        self._K_tgt = tf.constant(self.K_inv_tgt)
        self._K_src = tf.constant(self.K_inv_src)

    def loss(self):
        XT_pred = tf.matmul(self._UT, self._VT, transpose_b=True)
        XS_pred = tf.matmul(self._US, self._VS, transpose_b=True)
        
        tgt_mask = tf.math.logical_not(tf.is_nan(self._XT))
        tgt_err = tf.boolean_mask(self._XT - XT_pred, tgt_mask)
        tgt_sq_err = 0.5 * tf.square(tf.norm(tgt_err))

        src_mask = tf.math.logical_not(tf.is_nan(self._XS))
        src_err = tf.boolean_mask(self._XS - XS_pred, src_mask)
        src_sq_err = 0.5 * self.lambda_src * tf.square(tf.norm(src_err))


        reg_Vs = 0.5 * self.lambda_vs * tf.square(tf.norm(self._VS))
        reg_Vt  = 0.5 * self.lambda_v * tf.square(tf.norm(self._VT))

        reg_Us = 0.5 * self.lambda_us * \
                 tf.trace(tf.transpose(self._US) @ self._K_src @ self._US)
        reg_Ut  = 0.5 * self.lambda_u * \
                 tf.trace(tf.transpose(self._UT) @ self._K_tgt @ self._UT)

        # TODO:
        VtSVs = tf.matmul(tf.matmul(self._VT, self.sim_scores, transpose_a=True), self._VS)
        sim_sqr_err =  -tf.trace(VtSVs)
        sim_sqr_err_reg = 0.5 * self.lambda_sim * sim_sqr_err

        reg = reg_Vs + reg_Vt + reg_Us + reg_Ut
        return src_sq_err + tgt_sq_err + reg + sim_sqr_err_reg, \
               sim_sqr_err_reg, \
               tgt_sq_err, \
               src_sq_err, \
               reg
    
    def get_params(self):
        attributes = [
            'lambda_sim',
            'lambda_src',
            'lambda_tgt_rl',
            'lambda_src_rl',
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
        ]
        params = {}
        for attr in attributes:
            params[attr] = getattr(self, attr)
        
        return params


class KXSMF_b(XSMF):
    def __init__(self, X_tgt, X_src,
                 sim_scores, L_tgt, L_src,
                 X_val=None,
                 lambda_tgt_rl=0.1, # tgt regularized laplacian lambda
                 lambda_src_rl=0.1, # src regularized laplacian lambda
                 lambda_b=0.1, # bias regularizer
                 lambda_u=0.1,
                 lambda_v=0.1,
                 lambda_us=0.1,
                 lambda_vs=0.1,
                 lambda_sim=0.1,
                 lambda_src=0.1,
                 rank=40,
                 max_iter=150,
                 lr=0.1,
                 report_every=-1,
                 verbosity=logging.INFO,
                 early_stop=10):
        super().__init__(
            X_tgt=X_tgt,
            X_src=X_src,
            sim_scores=sim_scores,
            X_val=X_val,
            lambda_sim=lambda_sim,
            lambda_src=lambda_src,
            lambda_u=lambda_u,
            lambda_v=lambda_v,
            lambda_us=lambda_us,
            lambda_vs=lambda_vs,
            rank=rank,
            max_iter=max_iter,
            lr=lr,
            report_every=report_every,
            verbosity=verbosity,
            early_stop=early_stop
        )

        self.lambda_b = lambda_b

        self.L_tgt = L_tgt # target laplacian
        self.L_src = L_src # source laplacian
        self.lambda_tgt_rl = lambda_tgt_rl # target regularized laplacian param
        self.lambda_src_rl = lambda_src_rl # source regularized laplacian param

        self.K_inv_tgt = self.lambda_tgt_rl * self.L_tgt +  np.eye(X_tgt.shape[0])
        self.K_inv_src = self.lambda_tgt_rl * self.L_src +  np.eye(X_src.shape[0])
        self._initialize_tf_kernel_vars()

        self.bias_tgt = np.nansum(X_tgt, axis=0, keepdims=True)
        self.bias_tgt = self.bias_tgt/ np.nansum(np.isnan(X_tgt), axis=0, keepdims=False)

        self.bias_src = np.nansum(X_src, axis=0, keepdims=True)
        self.bias_src = self.bias_src / np.nansum(np.isnan(X_src), axis=0, keepdims=False)
        self._initialize_tf_bias_vars()

    def _initialize_tf_kernel_vars(self):
        self._K_tgt = tf.constant(self.K_inv_tgt)
        self._K_src = tf.constant(self.K_inv_src)
        
    def _initialize_tf_bias_vars(self):
        self._b_tgt = tf.Variable(self.bias_tgt)
        self._b_src = tf.Variable(self.bias_src)

    def loss(self):
        XT_pred = tf.matmul(self._UT, self._VT, transpose_b=True) + self._b_tgt
        XS_pred = tf.matmul(self._US, self._VS, transpose_b=True) + self._b_src
        
        tgt_mask = tf.math.logical_not(tf.is_nan(self._XT))
        tgt_err = tf.boolean_mask(self._XT - XT_pred, tgt_mask)
        tgt_sq_err = 0.5 * tf.square(tf.norm(tgt_err))

        src_mask = tf.math.logical_not(tf.is_nan(self._XS))
        src_err = tf.boolean_mask(self._XS - XS_pred, src_mask)
        src_sq_err = 0.5 * self.lambda_src * tf.square(tf.norm(src_err))


        reg_Vs = 0.5 * self.lambda_vs * tf.square(tf.norm(self._VS))
        reg_Vt  = 0.5 * self.lambda_v * tf.square(tf.norm(self._VT))

        reg_Us = 0.5 * self.lambda_us * \
                 tf.trace(tf.transpose(self._US) @ self._K_src @ self._US)
        reg_Ut  = 0.5 * self.lambda_u * \
                 tf.trace(tf.transpose(self._UT) @ self._K_tgt @ self._UT)

        # TODO:
        VtSVs = tf.matmul(tf.matmul(self._VT, self.sim_scores, transpose_a=True), self._VS)
        sim_sqr_err =  -tf.trace(VtSVs)
        sim_sqr_err_reg = 0.5 * self.lambda_sim * sim_sqr_err

        bias_reg_mat = tf.matmul(tf.matmul(self._b_tgt, self.sim_scores, transpose_a=False), self._b_src, transpose_b=True)

        bias_reg = -0.5 * self.lambda_b * tf.trace(bias_reg_mat)

        reg = reg_Vs + reg_Vt + reg_Us + reg_Ut + bias_reg
        return src_sq_err + tgt_sq_err + reg + sim_sqr_err_reg, \
               sim_sqr_err_reg, \
               tgt_sq_err, \
               src_sq_err, \
               reg
    
    
    def fit(self):        
        loss, sim_sqr_diff, tgt_sq_err, src_sq_err, reg = self.loss()
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
                err, train_r2, err_sims, err_tgt, err_src, err_reg = sess.run([loss, train_r2_tf, sim_sqr_diff,  tgt_sq_err, src_sq_err, reg])
                self.train_loss_hist.append(err)
                self.train_r2_hist.append(train_r2)

                if self.X_val is not None:
                    val_r2 = sess.run(val_r2_tf)
                    self.val_r2_hist.append(val_r2)
                sess.run(train_step)

                # Reporting
                if self.report_every >= 1 and i % self.report_every == 0:
                    self.log.info("\titer={:5d} | Training   | Cost={:8.4f}".format(i, err))
                    self.log.info("\titer={:5d} | Training   | Sim Sqr Diff={:8.4f}".format(i, err_sims))
                    self.log.info("\titer={:5d} | Training   | tgt Sqr Diff={:8.4f}".format(i, err_tgt))
                    self.log.info("\titer={:5d} | Training   | src Sqr Diff={:8.4f}".format(i, err_src))
                    self.log.info("\titer={:5d} | Training   | reg Sqr Diff={:8.4f}".format(i, err_reg))

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
            Ut, Us, Vt, Vs = \
                sess.run([self._UT, self._US, self._VT, self._VS])
            
            bias_tgt, bias_src = \
                sess.run([self._b_tgt, self._b_src])
            
        self.XS_fitted = np.dot(Us, Vs.T) + bias_src
        self.X_fitted = np.dot(Ut, Vt.T) + bias_tgt


        # TODO: self.U, self.US, self.V, self.VS need to be reordered
        
        self.U = Ut
        self.V = Vt
        self.US = Us
        self.VS = Vs
        self.bias_src = bias_src
        self.bias_tgt = bias_tgt

        # assert np.allclose(self.X_fitted, np.dot(self.U, self.V.T))
        # assert np.allclose(self.XS_fitted, np.dot(self.US, self.VS.T))

        # Store how many iterations we ran for for convenience
        self.iters_ran = len(self.train_loss_hist)

        return self
    
    def tgt_r2_score(self, X):
        mask = ~tf.is_nan(X)
        XT_pred = tf.matmul(self._UT, self._VT, transpose_b=True) + self._b_tgt
        
        true_vals = tf.boolean_mask(X, mask)
        imputed_vals = tf.boolean_mask(XT_pred, mask)

        return self._tf_r2_score_helper(true_vals, imputed_vals)
    
    def get_params(self):
        attributes = [
            'lambda_sim',
            'lambda_src',
            'lambda_tgt_rl',
            'lambda_src_rl',
            'lambda_u',
            'lambda_v',
            'lambda_us',
            'lambda_vs',
            'lambda_b',
            'rank',
            'max_iter',
            'lr',
            'report_every',
            'verbosity',
            'early_stop',
        ]
        params = {}
        for attr in attributes:
            params[attr] = getattr(self, attr)
        
        return params

#### CLI 

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Main parser
    # Parser
    parser.add_argument('-tgt', '--target_gis', type=str, required=True)
    parser.add_argument('-src', '--source_gis', type=str, required=True)
    parser.add_argument('-sim', '--sim_scores', type=str, required=True)
    parser.add_argument('-of', '--results_output', type=str, required=True)
    parser.add_argument('-log', '--logfile', type=str, default='/tmp/log.txt')

    return parser.parse_args()

def normalize_sim_scores(sim_scores):
    sim_scores[:, 2] = sim_scores[:, 2] / np.max(sim_scores[:, 2])
    return sim_scores

def main():
    pass
 
if __name__ == "__main__":
    main()
