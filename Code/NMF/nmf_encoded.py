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
from tqdm import trange
from ml_base import norm,nndsvda_init,getOptimizer,activateVariable
from ml_base import softplus_inverse,_beta_divergence,np_softplus

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
	
	def transform_noY(self,X):
		Scores_latent = np.dot(X,self.A_enc)+self.B_enc
		Scores = np_softplus(Scores_latent)
		return Scores
	
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
							mu=1.0,reg=.01,batchSize=100):
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
		self.batchSize = int(batchSize)

		self.creationDate = dt.now()
		self.version = version

		self.mu = float(mu)
		self.reg = float(reg)
	
	def fit_transform(self,X,y,W_init):
		
		################################
		# Declare our latent variables #
		################################
		if self.decoderActiv == 'softplus':
			W_l = softplus_inverse(W_init)
			W_r = tf.Variable(W_l.astype(np.float32))
		else:
			W_r = tf.Variable(W_init.astype(np.float32))

		phi =tf.Variable(np.zeros((self.n_components,1)).astype(np.float32))
		B = tf.Variable(.1*rand.randn(1).astype(np.float32))

		#print(tf.debugging.set_log_device_placement(True))

		#######################
		# Change X to float32 #
		#######################
		X = X.astype(np.float32)
		y = y.astype(np.float32)
		N,p = X.shape

		A_enci = 1/np.sqrt(p)*rand.randn(p,
							self.n_components).astype(np.float32)
		B_enci = 1/np.sqrt(self.n_components)*rand.randn(self.n_components).astype(np.float32)
		A_enc = tf.Variable(A_enci)
		B_enc = tf.Variable(B_enci)

		#####################
		# Get the optimizer #
		#####################
		optimizer = getOptimizer(self.LR,self.trainingMethod)
		optimizer2 = getOptimizer(self.LR,self.trainingMethod)
		optimizer3 = getOptimizer(self.LR,self.trainingMethod)

		losses_gen = np.zeros(self.nIter)
		losses_sup = np.zeros(self.nIter)

		#trainable_variables = [W_r,A_enc,B_enc]
		trainable_variables = [W_r,phi,B,A_enc,B_enc]
		trainable_variables3 = [W_r,A_enc,B_enc]
		trainable_variables2 = [phi,B,A_enc,B_enc]
		print('Hello')

		for t in trange(2000):
			idx = rand.choice(N,size=self.batchSize,replace=False)
			X_batch = X[idx]
			with tf.GradientTape() as tape:
				#W = activateVariable(W_r,
				#			self.decoderActiv)
				W = tf.nn.softplus(W_r)
				#W = tf.nn.relu(W_r) + 1e-5
				#W = tf.math.l2_normalize(W_un,axis=1)*np.sqrt(p)
				reg_W = tf.reduce_mean(tf.square(W))
				S_latent = tf.matmul(X_batch,A_enc) + B_enc
				S = tf.nn.softplus(S_latent)
				reg_S = tf.reduce_mean(tf.square(S))
				X_recon = tf.matmul(S,W)
				loss_gen = tf.reduce_mean(tf.square(X_recon-X_batch))
				loss = loss_gen + reg_W + reg_S

			gradients3 = tape.gradient(loss,trainable_variables3)
			optimizer3.apply_gradients(zip(gradients3,
										trainable_variables3))

		############################
		# Actually train the model #
		############################
		for t in trange(self.nIter):
			idx = rand.choice(N,size=self.batchSize,replace=False)
			X_batch = X[idx]
			Y_batch = y[idx]
			with tf.GradientTape() as tape:
				#W = activateVariable(W_r,
				#			self.decoderActiv)
				W = tf.nn.softplus(W_r)
				#W = tf.nn.relu(W_r) + 1e-5
				#W = tf.math.l2_normalize(W_un,axis=1)*np.sqrt(p)
				reg_W = tf.reduce_mean(tf.square(W))
				S_latent = tf.matmul(X_batch,A_enc) + B_enc
				S = tf.nn.softplus(S_latent)
				reg_S = tf.reduce_mean(tf.square(S))
				X_recon = tf.matmul(S,W)
				loss_gen = tf.reduce_mean(tf.square(X_recon-X_batch))
				y_hat = tf.matmul(S,phi) + B
				ce = tf.nn.sigmoid_cross_entropy_with_logits(Y_batch,
											tf.squeeze(y_hat))
				loss_sup = tf.reduce_mean(ce)
				loss_reg = tf.nn.l2_loss(phi)
				loss = loss_gen + self.reg*loss_reg + self.mu*loss_sup + reg_W + reg_S
			if t == 0:
				print(np.mean((W_init-W.numpy())**2))
				X_r = X_recon.numpy()
				print(np.mean((X_r-X_batch)**2))

			gradients = tape.gradient(loss,trainable_variables)
			optimizer.apply_gradients(zip(gradients,
										trainable_variables))

			losses_gen[t] = loss_gen.numpy()
			losses_sup[t] = loss_sup.numpy()

			'''
			for i in range(1):
				idx = rand.choice(N,size=self.batchSize,replace=False)
				X_batch = X[idx]
				Y_batch = y[idx]
				with tf.GradientTape() as tape:
					S_latent = tf.matmul(X_batch,A_enc) + B_enc
					S = tf.nn.softplus(S_latent)
					y_hat = tf.matmul(S,phi) + B
					ce = tf.nn.sigmoid_cross_entropy_with_logits(Y_batch,
												tf.squeeze(y_hat))
					loss_sup = tf.reduce_mean(ce)
					loss_reg = tf.nn.l2_loss(phi)
					loss = self.reg*loss_reg + self.mu*loss_sup 

				gradients2 = tape.gradient(loss,trainable_variables2)
				optimizer2.apply_gradients(zip(gradients2,
											trainable_variables2))
			'''

		self.losses_gen = losses_gen
		self.losses_sup = losses_sup

		#W_f = activateVariable(W_r,self.decoderActiv)
		#W_ = tf.math.l2_normalize(W_f,axis=1)*np.sqrt(p)
		W_ = tf.nn.softplus(W_r)
		S_r = tf.matmul(X,A_enc) + B_enc
		S_f = activateVariable(S_r,self.decoderActiv)

		self.components_ = W_.numpy()
		Scores = S_f.numpy()
		self.phi_ = phi.numpy()
		self.b_ = B.numpy()
		self.A_enc = A_enc.numpy()
		self.B_enc = B_enc.numpy()
		self.W_ = W_
		self.reconstruction_err_ = _beta_divergence(X,Scores,
										self.components_,beta=self.beta)
		yh = np.dot(Scores,self.phi_) + self.b_
		fpr,tpr,_ = roc_curve(y,yh)
		self.sup_loss_ = auc(fpr,tpr)
		return Scores



