"""
Code taken from Network Guided Matrix Factorization project from

Marinka Zitnik's github.com/marinkaz/ngmc
"""

import numpy as np
from matrix_completion import BaseMatrixCompletion
from sklearn.metrics import r2_score

class NGMC(BaseMatrixCompletion):
    def __init__(self, rank, lambda_f=0.01, lambda_h=0.01, lambda_p=0.01,
                 alpha=0.1, alpha_p=0.001, max_iter=150, burnout=0, lr_decay=1.0,
                 callback=None,
                 verbose=False):
        self.rank = rank
        self.lambda_f = lambda_f
        self.lambda_h = lambda_h
        self.lambda_p = lambda_p
        self.alpha = alpha
        self.alpha_p = alpha_p
        self.max_iter = max_iter
        self.burnout = burnout
        self.lr_decay = lr_decay
        self.callback = callback
        self.verbose = verbose

        ### logging
        self.train_r2_hist = []
        self.val_r2_hist = []
    
    def fit(self, G, P=None, X_val=None):
        self.X = G
        self.X_val = X_val
        self.P = P
        _val_callback  = None

        if X_val is not None:
            _val_callback = self._val_callback

        Ps = np.asarray([self.P]) if self.P is not None else self.P
        self._ngmc = _NGMC(self.X, 
                           self.rank, 
                           P=Ps, 
                           lambda_u=self.lambda_f, 
                           lambda_v=self.lambda_h, 
                           lambda_p=self.lambda_p, 
                           alpha=self.alpha, 
                           alpha_p=self.alpha_p, 
                           max_iter=self.max_iter, 
                           burnout=self.burnout,
                           lr_decay=self.lr_decay,
                           callback=self._val_callback if X_val is not None else None)
        self._ngmc.fit(verbose=self.verbose)
        
        self.iters_ran = len(self.train_r2_hist)
        self.X_fitted = self._ngmc.imputed()
        return self

    def _val_callback(self, ngmc):
        X_fitted = ngmc.imputed()
        train_mask = ~np.isnan(self.X)
        val_mask = ~np.isnan(self.X_val)
        tr_score = r2_score(self.X[train_mask], X_fitted[train_mask])
        val_score = r2_score(self.X_val[val_mask], X_fitted[val_mask])

        self.train_r2_hist.append(tr_score)
        self.val_r2_hist.append(val_score)
    
    def training_curve(self):
        return dict(train=self.train_r2_hist, val=self.val_r2_hist)
    

class _NGMC(object):
    """
    Class taken directly from github.com/marinkaz/ngmc/ngmc.py
    """
    def __init__(self, G, c, P=None, lambda_u=0.01, lambda_v=0.01, lambda_p=0.01,
                 alpha=0.1, alpha_p=0.001, max_iter=150, burnout=0, lr_decay=1.0,
                 callback=None):
        self.G = G
        self.Gma = np.ma.masked_array(self.G, np.isnan(self.G))
        self.P = P
        self.c = c
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_p = lambda_p
        self.alpha = alpha
        self.alpha_p = alpha_p
        self.max_iter = max_iter
        self.burnout = burnout
        self.lr_decay = lr_decay
        self.callback = callback

        self.F = None
        self.H = None
        self.W = None

    def _initialize(self):
        self.F = np.random.rand(self.c, self.G.shape[0])
        self.H = np.random.rand(self.c, self.G.shape[1])
        if self.P is not None:
            n_net = len(self.P)
            self.W = 1./n_net*np.ones((self.G.shape[0], n_net))
    def _g(self, x):
        return 1./(1.+np.exp(-0.5*x))

    def _g_prime(self, x):
        return 0.5*np.exp(-0.5*x)/(1.+np.exp(-0.5*x))**2

    def _F_prime(self, itr):
        G_hat = np.dot(self.F.T, self.H)
        G_hat_g_prime = self._g_prime(G_hat)
        G_hat_g = self._g(G_hat)
        der_f = np.ma.multiply(G_hat_g_prime, G_hat_g-self.Gma).T
        F_prime = np.ma.dot(self.H, der_f)
        if itr > self.burnout and self.P is not None:
            P_multi = np.zeros(self.P[0].shape)
            for i in range(len(self.P)):
                Wi = self.W[:, i].reshape((self.W.shape[0], 1))
                Pi = self.P[i]
                Wit = np.tile(Wi, (1, Wi.shape[0]))
                P_multi += np.multiply(Pi, Wit)
            P2 = self.F-np.dot(self.F, P_multi.T)
            P3 = np.dot(self.F-np.dot(self.F, P_multi.T), P_multi)
            return F_prime+self.lambda_u*self.F+self.lambda_p*P2-self.lambda_p*P3
        else:
            return F_prime+self.lambda_u*self.F

    def _W_prime(self, itr):
        W_prime = np.zeros(self.W.shape)
        if itr < self.burnout:
            return W_prime
        for i in range(len(self.P)):
            tmp = np.dot(self.F, self.P[i].T)
            T1 = np.zeros(self.W.shape[0])
            T2 = np.zeros(self.W.shape[0])
            for u in range(W_prime.shape[0]):
                T1[u] = np.dot(self.F[:, u], tmp[:, u])
                T2[u] = self.W[u, i]*np.dot(tmp[:, u], tmp[:, u])

            tmp1 = np.zeros(self.F.shape)
            for p in range(len(self.P)):
                if p == i: continue
                tmp11 = np.dot(self.F, self.P[p].T)
                Wp = self.W[:, p].reshape((self.W.shape[0], 1))
                Wpt = np.tile(Wp, (1, self.F.shape[0])).T
                tmp1 += np.multiply(Wpt, tmp11)

            T3 = np.zeros(self.W.shape[0])
            for u in range(self.W.shape[0]):
                T3[u] = np.dot(tmp[:, u], tmp1[:, u])

            W_prime[:, i] = -self.lambda_p*T1+self.lambda_p*T2+self.lambda_p/2.*T3
        return W_prime

    def _H_prime(self):
        G_hat = np.dot(self.F.T, self.H)
        G_hat_g_prime = self._g_prime(G_hat)
        G_hat_g = self._g(G_hat)
        H_prime = np.ma.dot(self.F, np.ma.multiply(G_hat_g_prime, G_hat_g-self.Gma))
        return H_prime+self.lambda_v*self.H

    def fit(self, verbose=True):
        self._initialize()
        err = [1e10, 1e10]
        nrm = [1e10, 1e9]
        for itr in range(self.max_iter):
            if err[-1] > err[-2]: #and nrm[-1] < nrm[-2]:
                break
            err[-2] = err[-1]
            nrm[-2] = nrm[-1]
            F_prime = self._F_prime(itr)
            self.F = self.F-self.alpha*F_prime
            H_prime = self._H_prime()
            self.H = self.H-self.alpha*H_prime
            if self.P is not None:
                W_prime = self._W_prime(itr)
                self.W = self.W-self.alpha_p*W_prime
            G_hat = self.imputed()
            sq = np.ma.multiply(self.Gma-G_hat, self.Gma-G_hat)
            fro = np.sqrt(np.nansum(sq))
            nrmse = np.sqrt(np.ma.mean(sq)/np.ma.var(self.Gma))
            if verbose:
                print("Iter: %d: Fro(G-G_hat)[known] = %5.4f" % (itr, fro))
                print("Iter: %d: NRMSE(G, G_hat)[known] = %5.4f" % (itr, nrmse))
            nrm[-1] = nrmse
            err[-1] = fro
            self.alpha *= self.lr_decay
            if self.callback:
                self.callback(self)
        return self.F, self.H, self.W

    def imputed(self):
        G_hat = np.dot(self.F.T, self.H)
        return self._g(G_hat)