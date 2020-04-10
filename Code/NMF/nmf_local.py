'''
Creator:
    Austin "The Man" Talbot
Creation Date:
    11/16/2019
Version history
---------------
Version 1.0
Objects
-------
sNMF_L1
sNMF_blessed
References
----------
https://www.tensorflow.org/

https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html#sphx-glr-auto-examples-decomposition-plot-faces-decomposition-py
'''
import numpy as np
import numpy.random as rand
import sys,os
import tensorflow as tf
from tensorflow import keras
from datetime import datetime as dt
import numpy.linalg as la
import pickle
import time
from sklearn.utils.extmath import randomized_svd,squared_norm
from sklearn import decomposition as dp
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.metrics import roc_curve,auc
#from tqdm import range
from ml_base import norm,nndsvda_init,getOptimizer,activateVariable
from ml_base import softplus_inverse,_beta_divergence

version = '1.0'

class sNMF_base(object):

	def __init__(self,n_components,nIter=50000,LR=1e-5,name='NMF_g',
							dirName='./tmp',device=0,gpuMem=1024,
							decoderActiv='softplus',factorActiv='softplus',
							trainingMethod='Nadam',beta='frobenius',mu=1.0):
		self.n_components = int(n_components)
		self.nIter = int(nIter)
		self.LR = float(LR)
		self.beta = beta

		self.device = int(device)
		self.gpuMem = int(gpuMem)

		self.name = str(name)
		self.dirName = str(dirName)

		self.trainingMethod = str(trainingMethod)
		self.factorActiv = factorActiv
		self.decoderActiv = decoderActiv

		self.creationDate = dt.now()
		self.version = version

		self.mu = float(mu)
	
	def transform_Y(self,X,y):
		######################################
		# Initialize the scores and features #
		######################################
		S_init,_ = nndsvda_init(X,self.n_components)

		################################
		# Declare our latent variables #
		################################
		W = tf.constant(self.components_.astype(np.float32))
		Phi_ = self.phi_
		B = self.b_
		if self.factorActiv == 'softplus':
			S_l = softplus_inverse(S_init)
			S_r = tf.Variable(S_l.astype(np.float32))
		else:
			S_r = tf.Variable(S_init.astype(np.float32))

		#######################
		# Change X to float32 #
		#######################
		X = X.astype(np.float32)
		y = y.astype(np.float32)

		#####################
		# Get the optimizer #
		#####################
		optimizer = getOptimizer(self.LR,self.trainingMethod)

		trainable_variables = [S_r]

		losses = np.zeros(self.nIter)

		############################
		# Actually train the model #
		############################
		for t in range(self.nIter):

			with tf.GradientTape() as tape:
				S = activateVariable(S_r,self.factorActiv)
				X_recon = tf.matmul(S,W)
				y_hat = tf.matmul(S,Phi_) + B
				ce= tf.nn.sigmoid_cross_entropy_with_logits(y,
										tf.squeeze(y_hat))
				loss_sup = tf.reduce_mean(ce)
				loss = tf.nn.l2_loss(X-X_recon)+self.mu*loss_sup

			losses[t] = loss.numpy()

			gradients = tape.gradient(loss,trainable_variables)
			optimizer.apply_gradients(zip(gradients,trainable_variables))

		S_f = activateVariable(S_r,self.factorActiv)

		Scores = S_f.numpy()
		return Scores,losses
	
	def transform_noY(self,X):
		mod = dp.NMF(self.n_components)
		mod.fit(X)
		mod.components_ = self.components_.astype(np.float64)
		S = mod.transform(X)
		return S

	def transform_noY_old(self,X):
		######################################
		# Initialize the scores and features #
		######################################
		S_init,_ = nndsvda_init(X,self.n_components)

		################################
		# Declare our latent variables #
		################################
		W = tf.constant(self.components_.astype(np.float32))
		if self.factorActiv == 'softplus':
			S_l = softplus_inverse(S_init)
			S_r = tf.Variable(S_l.astype(np.float32))
		else:
			S_r = tf.Variable(S_init.astype(np.float32))

		#######################
		# Change X to float32 #
		#######################
		X = X.astype(np.float32)

		#####################
		# Get the optimizer #
		#####################
		optimizer = getOptimizer(200*self.LR,self.trainingMethod)

		trainable_variables = [S_r]
		
		losses = np.zeros(1000)

		############################
		# Actually train the model #
		############################
		for t in range(1000):

			with tf.GradientTape() as tape:
				S = activateVariable(S_r,self.factorActiv)
				X_recon = tf.matmul(S,W)
				loss = tf.nn.l2_loss(X-X_recon)

			losses[t] = loss.numpy()

			gradients = tape.gradient(loss,trainable_variables)
			optimizer.apply_gradients(zip(gradients,trainable_variables))

		S_f = activateVariable(S_r,self.factorActiv)

		Scores = S_f.numpy()

		return Scores,losses
	
	def saveModel(self,saveName):
		myDict = {'model':self}
		pickle.dump(myDict,open(saveName,'wb'))
	
	def saveComponents(self,saveName,method='matlab'):
		if method == 'matlab':
			myDict = {'components':self.components_}
			scipy.io.savemat(saveName,myDict)
		elif method == 'csv':
			np.savetxt(saveName,self.components_,fmt='%0.8f',delimiter=',')
		else:
			print('Unrecognized save method %s'%method)
	

class sNMF_L1(sNMF_base):
	
	def __init__(self,n_components,nIter=50000,LR=1e-3,name='NMF_g',
							dirName='./tmp',device=0,gpuMem=1024,
							decoderActiv='softplus',factorActiv='softplus',
							trainingMethod='Nadam',beta='frobenius',
							mu=1.0,reg=.01):
		self.n_components = int(n_components)
		self.nIter = int(nIter)
		self.LR = float(LR)
		self.beta = beta

		self.device = int(device)
		self.gpuMem = int(gpuMem)

		self.name = str(name)
		self.dirName = str(dirName)

		self.trainingMethod = str(trainingMethod)
		self.factorActiv = factorActiv
		self.decoderActiv = decoderActiv

		self.creationDate = dt.now()
		self.version = version

		self.mu = float(mu)
		self.reg = float(reg)
	
	def fit_transform(self,X,y,S_init=None,W_init=None):
		
		######################################
		# Initialize the scores and features #
		######################################
		if S_init is None:
			S_init,W_init = nndsvda_init(X,self.n_components)
		else:
			pass

		################################
		# Declare our latent variables #
		################################
		if self.decoderActiv == 'softplus':
			W_l = softplus_inverse(W_init)
			W_r = tf.Variable(W_l.astype(np.float32))
		else:
			W_r = tf.Variable(W_init.astype(np.float32))
		if self.factorActiv == 'softplus':
			S_l = softplus_inverse(S_init)
			S_r = tf.Variable(S_l.astype(np.float32))
		else:
			S_r = tf.Variable(S_init.astype(np.float32))
		mod_lr = LogReg()
		mod_lr.fit(S_init,y)
		cinit = np.zeros((self.n_components,1))
		cinit[:,0] = mod_lr.coef_
		phi =tf.Variable(cinit.astype(np.float32))
		binit = rand.randn(1)
		binit[0] = mod_lr.intercept_
		B = tf.Variable(binit.astype(np.float32))

		X_recon_orig = np.dot(S_init,W_init)

		#######################
		# Change X to float32 #
		#######################
		X = X.astype(np.float32)
		y = y.astype(np.float32)
		N,p = X.shape

		#####################
		# Get the optimizer #
		#####################
		optimizer = getOptimizer(self.LR,self.trainingMethod)
		optimizer2 = getOptimizer(.1*self.LR,self.trainingMethod)

		losses_gen = np.zeros(self.nIter)
		losses_sup = np.zeros(self.nIter)

		trainable_variables = [W_r,phi,B,S_r]
		trainable_variables2 = [S_r]

		############################
		# Actually train the model #
		############################
		startTime = time.time()
		for t in range(self.nIter):
			
			nInner = 1
			#for i in range(nInner):
			if 1 == 1:
				with tf.GradientTape() as tape:
					W_un = activateVariable(W_r,
								self.decoderActiv)
					W = tf.math.l2_normalize(W_un,axis=1)*np.sqrt(p)
					S = activateVariable(S_r,self.factorActiv)
					X_recon = tf.matmul(S,W)
					loss_gen = tf.nn.l2_loss(X-X_recon)/N
					y_hat = tf.matmul(S,phi) + B
					ce = tf.nn.sigmoid_cross_entropy_with_logits(y,
												tf.squeeze(y_hat))
					loss_sup = tf.reduce_mean(ce)
					loss_reg = tf.nn.l2_loss(phi)
					loss = loss_gen + self.reg*loss_reg + self.mu*loss_sup 
				gradients = tape.gradient(loss,trainable_variables)
				optimizer.apply_gradients(zip(gradients,
											trainable_variables))

			losses_gen[t] = loss_gen.numpy()
			losses_sup[t] = loss_sup.numpy()

			if t % 1000 == 0:
				el = time.time() - startTime
				print(t,el,losses_sup[t])
			

		self.losses_gen = losses_gen
		self.losses_sup = losses_sup

		W_f = activateVariable(W_r,self.decoderActiv)
		W_ = tf.math.l2_normalize(W_f,axis=1)*np.sqrt(p)
		S_f = activateVariable(S_r,self.decoderActiv)

		self.components_ = W_.numpy()
		Scores = S_f.numpy()
		self.phi_ = phi.numpy()
		self.b_ = B.numpy()
		self.reconstruction_err_ = _beta_divergence(X,Scores,
										self.components_,beta=self.beta)
		yh = y_hat.numpy()
		fpr,tpr,_ = roc_curve(y,yh)
		self.sup_loss_ = auc(fpr,tpr)
		return Scores


