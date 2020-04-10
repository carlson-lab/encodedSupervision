'''la.norm(comp1)
Creator:
    Austin "The Man" Talbot
Creation Date:
    8/17/19
Version history
---------------
Version 1.0
Objects
-------

'''
from math import log,sqrt
import numbers

import numpy as np
from scipy import linalg as la
from scipy.special import gammaln
from scipy.sparse import issparse
from scipy.sparse.linalg import svds
from numpy import random as rand

from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import randomized_svd,fast_logdet,svd_flip
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_is_fitted
from sklearn import decomposition as dp

from datetime import datetime as dt
import os,sys,time
from scipy.io import savemat

version = '1.0'

def sortedEig(E):
	eigenValues,eigenVectors = la.eig(E)
	idx = eigenValues.argsort()[::-1]
	eigenValues = eigenValues[idx]
	eigenVectors = eigenVectors[:,idx]
	return eigenValues,eigenVectors

class spca_base(object):

	def _createDefaultName(self,defaultName,fileName):
		'''Small auxiliary method for creating a customized name
		'''
		if fileName is None:
			return defaultName
		else:
			return fileName
	

class spca_enc(spca_base):
	r'''Encoded supervision principal component analysis

	Parameters
	----------
	n_components : int
		Number of components to keep

	Attributes
	----------

	References
	----------
	For probabilistic PCA see:
	Tipping, M. E., and Bishop, C. M. (1999). "Probabilistic principal
	component analysis". Journal of the Royal Statistical Society:
	Series B (Statistical Methodology), 61(3), 611-622.


	Examples
	--------

	'''
	def __init__(self,n_components,name='spca_enc',lamb=.1):
		self.n_components = int(n_components)
		self.name = str(name)
		self.lamb = float(lamb)
			
		self.creationDate = dt.now()
		self.version = version
	
	def fit(self,X,Y):
		'''Fit the model with X.

		Parameters
		----------

		Returns
		-------
		
		'''
		######################
		# Input verification #
		######################
		X = check_array(X,dtype=[np.float64,np.float32],ensure_2d=True)
		Y = check_array(Y,dtype=[np.float64,np.float32],ensure_2d=True)


		N,self.p = X.shape
		_,self.q = Y.shape
		p,q = self.p,self.q
		if self.n_components>=self.p:
			raise ValueError("Too many factors %d for data of dimension %d"%(int(self.n_components),int(self.p)))

		X = X.T
		Y = Y.T

		XXT = np.dot(X,X.T)/N 
		XYT = np.dot(X,Y.T)/N
		YXT = np.dot(Y,X.T)/N

		#XXTi = la.inv(XXT+self.tik*np.eye(p))
		XXTi = la.inv(XXT)
		back = la.solve(XXT,XYT)
		YYT = np.dot(YXT,back)

		covMat = np.zeros((p+q,p+q))
		covMat[:p,:p] = XXT
		covMat[:p,p:] = self.lamb*XYT
		covMat[p:,:p] = YXT
		covMat[p:,p:] = self.lamb*YYT
		covMat = covMat

		eigenValues,eigenVectors = sortedEig(covMat)
		eigenValues = np.real(eigenValues)
		eigenVectors = np.real(eigenVectors)
		self.ev_ = eigenValues
		self.eVec_ = eigenVectors
		self.W_ = eigenVectors[:self.p,:self.n_components]
		self.D_ = eigenVectors[self.p:,:self.n_components]
		self.B_ = covMat

		#Now we have to learn A
		WTWmuDTD = (np.dot(self.W_.T,self.W_) + 
							self.lamb*np.dot(self.D_.T,self.D_))
		WTWi = la.inv(WTWmuDTD)
		DTY = self.lamb*np.dot(self.D_.T,Y)
		YXT = np.dot(DTY,X.T)/N
		remainder = self.W_.T + np.dot(YXT,XXTi)
		self.A_ = WTWi.dot(remainder)
		self.cov = covMat

	def transform(self,X):
		'''Fit the model with X.

		Parameters
		----------

		Returns
		-------
		
		'''
		X = check_array(X)


		N,_ = X.shape

		mul = np.dot(X,self.W_)
		WTW = np.dot(self.W_.T,self.W_)
		WTWi = la.inv(WTW)
		coord = np.dot(mul,WTWi)
		return coord
	
	def project(self,X):
		S = np.dot(X,self.A_.T)
		return S
	
	def predict(self,X):
		S = self.project(X)
		preds = np.dot(np.real(S),np.real(self.D_.T))
		return preds
	
	def reconstruct(self,X):
		S = self.project(X)
		X_recon = np.dot(np.real(S),np.real(self.W_.T))
		return X_recon

class spca_local(object):
	r'''Encoded supervision principal component analysis

	Parameters
	----------
	n_components : int
		Number of components to keep

	Attributes
	----------

	References
	----------
	For probabilistic PCA see:
	Tipping, M. E., and Bishop, C. M. (1999). "Probabilistic principal
	component analysis". Journal of the Royal Statistical Society:
	Series B (Statistical Methodology), 61(3), 611-622.


	Examples
	--------

	'''
	def __init__(self,n_components,name='spca_enc',lamb=.1):
		self.n_components = int(n_components)
		self.name = str(name)
		self.lamb = float(lamb)
			
		self.creationDate = dt.now()
		self.version = version
	
	def fit(self,X,Y):
		'''Fit the model with X.

		Parameters
		----------

		Returns
		-------
		
		'''
		######################
		# Input verification #
		######################
		X = check_array(X,dtype=[np.float64,np.float32],ensure_2d=True)
		Y = check_array(Y,dtype=[np.float64,np.float32],ensure_2d=True)


		N,self.p = X.shape
		_,self.q = Y.shape
		p,q = self.p,self.q
		if self.n_components>=self.p:
			raise ValueError("Too many factors %d for data of dimension %d"%(int(self.n_components),int(self.p)))

		X = X.T
		Y = Y.T

		XXT = np.dot(X,X.T)/N
		XYT = np.dot(X,Y.T)/N
		YXT = np.dot(Y,X.T)/N
		YYT = np.dot(Y,Y.T)/N


		covMat = np.zeros((p+q,p+q))
		covMat[:p,:p] = XXT
		covMat[:p,p:] = self.lamb*XYT
		covMat[p:,:p] = YXT
		covMat[p:,p:] = self.lamb*YYT
		covMat = covMat# + .1*np.eye(p+q)
		self.cov = covMat

		eigenValues,eigenVectors = sortedEig(covMat)
		self.ev_ = eigenValues
		self.eVec_ = eigenVectors
		self.W_ = eigenVectors[:self.p,:self.n_components]
		self.D_ = eigenVectors[self.p:,:self.n_components]
		self.B_ = covMat

		#Used for the transform method
		WTW = np.dot(self.W_.T,self.W_)
		DTD = np.dot(self.D_.T,self.D_)
		self.WTWmuDTD = WTW + self.lamb*DTD

	def project(self,X):
		'''Fit the model with X.

		Parameters
		----------

		Returns
		-------
		
		'''
		X = check_array(X)
		WTW = np.dot(self.W_.T,self.W_)
		WTX = np.dot(X,self.W_).T
		scores = la.solve(WTW,WTX)
		return scores.T
		#return np.dot(X,self.W_)
	
	def transform(self,X,Y):
		X = X.T
		Y = Y.T
		WTX = np.dot(self.W_.T,X)
		DTY = np.dot(self.D_.T,Y)
		right = WTX + self.lamb*DTY
		S = la.solve(self.WTWmuDTD,right)
		return S.T
	
	def predict(self,X):
		S = self.project(X)
		preds = np.dot(np.real(S),np.real(self.D_.T))
		return preds
	
	def reconstruct(self,X):
		S = self.project(X)
		X_recon = np.dot(np.real(S),np.real(self.W_.T))
		return X_recon
	
	def predict_fullyObserved(self,X,Y):
		S = self.transform(X,Y)
		X_recon = np.dot(np.real(S),np.real(self.D_.T))
		return X_recon

	def reconstruct_fullyObserved(self,X,Y):
		S = self.transform(X,Y)
		preds = np.dot(np.real(S),np.real(self.W_.T))
		return preds







