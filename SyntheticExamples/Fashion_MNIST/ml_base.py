'''
Creator:
	Austin "The Man" Talbot
Creation Date:
	11/16/2019
Version history
---------------
Version 1.0
Methods
-------
norm
	Gets a vector norm
nndsvda_init
	Initializes a non-negative matrix factorization. Copied straight from 
	sklearn.
References
----------
https://www.tensorflow.org/

https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html#sphx-glr-auto-examples-decomposition-plot-faces-decomposition-py
'''
import numpy as np
from numpy import random as rand
from sklearn.utils.extmath import randomized_svd,squared_norm
from sklearn import decomposition as dp
import tensorflow as tf
from tensorflow import keras 
from numpy import sqrt
import numbers
import scipy.sparse as sp

def np_softplus(x):
	return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x,0)

def norm(x):
	"""Dot product-based Euclidean norm implementation
	See: http://fseoane.net/blog/2011/computing-the-vector-norm/
	Parameters
	----------
	x : array-like
		Vector for which to compute the norm
	"""
	return sqrt(squared_norm(x))

def nndsvda_init(X,n_components,eps=1e-6,random_state=None):
	U,S,V = randomized_svd(X,n_components,random_state=random_state)
	W,H = np.zeros(U.shape),np.zeros(V.shape)

	# The leading singular triplet is non-negative
	# so it can be used as is for initialization.
	W[:,0] = np.sqrt(S[0]) * np.abs(U[:,0])
	H[0,:] = np.sqrt(S[0]) * np.abs(V[0,:])

	for j in range(1, n_components):
		x,y = U[:,j],V[j,:]

		# extract positive and negative parts of column vectors
		x_p,y_p = np.maximum(x,0),np.maximum(y,0)
		x_n,y_n = np.abs(np.minimum(x,0)),np.abs(np.minimum(y,0))

		# and their norms
		x_p_nrm,y_p_nrm = norm(x_p),norm(y_p)
		x_n_nrm,y_n_nrm = norm(x_n),norm(y_n)

		m_p,m_n = x_p_nrm*y_p_nrm,x_n_nrm*y_n_nrm

		# choose update
		if m_p > m_n:
			u = x_p/x_p_nrm
			v = y_p/y_p_nrm
			sigma = m_p
		else:
			u = x_n/x_n_nrm
			v = y_n/y_n_nrm
			sigma = m_n

		lbd = np.sqrt(S[j]*sigma)
		W[:,j] = lbd*u
		H[j,:] = lbd*v

	W[W<eps] = 0
	H[H<eps] = 0

	return W, H

def softplus_inverse(mat):
	return np.log(np.exp(mat) - 1 + 1e-8)

def getOptimizer(LR,method):
	if method == 'SGD':
		optimizer = keras.optimizers.SGD(learning_rate=LR)
	elif method == 'Momentum':
		optimizer = keras.optimizers.SGD(learning_rate=LR,momentum=0.9)
	elif method == 'Adam':
		optimizer = keras.optimizers.Adam(learning_rate=LR)
	elif method == 'Nadam':
		optimizer = keras.optimizers.Nadam(learning_rate=LR)
	else:
		print('Unrecognized method : %s'%method)
		optimizer = None
	
	return optimizer
	
def activateVariable(var,method):
	if method == 'softplus':
		return tf.nn.softplus(var)
	else:
		return tf.nn.relu(var)


def _beta_loss_to_float(beta_loss):
    """Convert string beta_loss to float"""
    allowed_beta_loss = {'frobenius': 2,
                         'kullback-leibler': 1,
                         'itakura-saito': 0}
    if isinstance(beta_loss, str) and beta_loss in allowed_beta_loss:
        beta_loss = allowed_beta_loss[beta_loss]

    if not isinstance(beta_loss, numbers.Number):
        raise ValueError('Invalid beta_loss parameter: got %r instead '
                         'of one of %r, or a float.' %
                         (beta_loss, allowed_beta_loss.keys()))
    return beta_loss

def _beta_divergence(X, W, H, beta, square_root=False):
    """Compute the beta-divergence of X and dot(W, H).
    Parameters
    ----------
    X : float or array-like, shape (n_samples, n_features)
    W : float or dense array-like, shape (n_samples, n_components)
    H : float or dense array-like, shape (n_components, n_features)
    beta : float, string in {'frobenius', 'kullback-leibler', 'itakura-saito'}
        Parameter of the beta-divergence.
        If beta == 2, this is half the Frobenius *squared* norm.
        If beta == 1, this is the generalized Kullback-Leibler divergence.
        If beta == 0, this is the Itakura-Saito divergence.
        Else, this is the general beta-divergence.
    square_root : boolean, default False
        If True, return np.sqrt(2 * res)
        For beta == 2, it corresponds to the Frobenius norm.
    Returns
    -------
        res : float
            Beta divergence of X and np.dot(X, H)
    """
    beta = _beta_loss_to_float(beta)

    # The method can be called with scalars
    if not sp.issparse(X):
        X = np.atleast_2d(X)
    W = np.atleast_2d(W)
    H = np.atleast_2d(H)

    # Frobenius norm
    if beta == 2:
        # Avoid the creation of the dense np.dot(W, H) if X is sparse.
        if sp.issparse(X):
            norm_X = np.dot(X.data, X.data)
            norm_WH = trace_dot(np.dot(np.dot(W.T, W), H), H)
            cross_prod = trace_dot((X * H.T), W)
            res = (norm_X + norm_WH - 2. * cross_prod) / 2.
        else:
            res = squared_norm(X - np.dot(W, H)) / 2.

        if square_root:
            return np.sqrt(res * 2)
        else:
            return res

    if sp.issparse(X):
        # compute np.dot(W, H) only where X is nonzero
        WH_data = _special_sparse_dot(W, H, X).data
        X_data = X.data
    else:
        WH = np.dot(W, H)
        WH_data = WH.ravel()
        X_data = X.ravel()

    # do not affect the zeros: here 0 ** (-1) = 0 and not infinity
    indices = X_data > EPSILON
    WH_data = WH_data[indices]
    X_data = X_data[indices]

    # used to avoid division by zero
    WH_data[WH_data == 0] = EPSILON

    # generalized Kullback-Leibler divergence
    if beta == 1:
        # fast and memory efficient computation of np.sum(np.dot(W, H))
        sum_WH = np.dot(np.sum(W, axis=0), np.sum(H, axis=1))
        # computes np.sum(X * log(X / WH)) only where X is nonzero
        div = X_data / WH_data
        res = np.dot(X_data, np.log(div))
        # add full np.sum(np.dot(W, H)) - np.sum(X)
        res += sum_WH - X_data.sum()

    # Itakura-Saito divergence
    elif beta == 0:
        div = X_data / WH_data
        res = np.sum(div) - np.product(X.shape) - np.sum(np.log(div))

    # beta-divergence, beta not in (0, 1, 2)
    else:
        if sp.issparse(X):
            # slow loop, but memory efficient computation of :
            # np.sum(np.dot(W, H) ** beta)
            sum_WH_beta = 0
            for i in range(X.shape[1]):
                sum_WH_beta += np.sum(np.dot(W, H[:, i]) ** beta)

        else:
            sum_WH_beta = np.sum(WH ** beta)

        sum_X_WH = np.dot(X_data, WH_data ** (beta - 1))
        res = (X_data ** beta).sum() - beta * sum_X_WH
        res += sum_WH_beta * (beta - 1)
        res /= beta * (beta - 1)

    if square_root:
        return np.sqrt(2 * res)
    else:
        return res


