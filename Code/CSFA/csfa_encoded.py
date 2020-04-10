'''
csfa
Creator:
	Austin "The Man" Talbot
Creation Date:
	1/1/19
Version history
---------------
Version 1.1-1.7
	Only God Himself knows
Version 1.8, 2/20/19:
	Added hierarchical structure with CSFA_base. Also fixed typos and added
	flexible input shape for data
Version 1.9, 2/20/19:
	Added multi-layer perceptron supervision
Version 1.10, 2/20/19:
	Split off multi-layer perceptron, modified CSFA_base
Version 1.11, 3/4/19
	Documentation as well as clearer options for encoder
Version 1.12, 3/8/19
	Bugs fixed as well as issues with purely MLP
Version 1.13, 3/19/19
	Modified the learning rate defaults to better values. Normally
	not worthy of a new version but this one actually matters.
Version 1.14, 3/24/19
	Now we need distinct learning rates depending on global features. 
	Also added different iteration amouts for encoder vs global. 
	Finally added saving abilities
Version 1.15 3/25/19
	Added the weighting for the supervision as well as an ability to 
	scale down the GPU usage.
Version 1.16 4/10/19
	Added the proper batchnorm because Tensorflow is f***** stupid
Version 1.17 4/11/19
	Added the generative weights 
Version 1.18 4/24/19
	Made ability to supervise on group
Version 1.19 5/10/19
	Added another stupid thing to batchnorm because Tensorflow is f******
	stupid. Also made a UKUnorm method
Objects
-------
CSFA_base
	The base encoded CSFA model. This has basic operations such as get 
	parameters, set parameters, create tensors to compute log-likelihood
	and optimizer definitions. Used for inheritance so all the boilerplate
	code goes in this object.
CSFA_encoded_dense
	Basic CSFA model with the scores replaced by an encoder. No supervision
	included. Various options for type of activation function number
	of iterations to optimize learning rate etc. Literally most of the 
	code to implement this put in CSFA_base.
dCSFA_encoded_dense
	Inital supervised CSFA model where the first n factors are blessed to
	be predictive. Cross-entropy loss with a reverse annealed supervision
	weight. Can choose start strength increment and max sup strength. The
	loss is (1-alpha)NLL + alpha*pred to avoid instability.
dCSFA_L1_encoded_dense
	Rather than choosing a number of factors to bless we instead make all
	factors predictive with a strong L1 penalty on the supervision 
	coefficients to ensure sparsity.
dCSFA_L1adaptive_encoded_dense
	Forces sparsity and adapts regularization
dCSFA_L1adaptive_encoded_dense_double
	Allows for supervision on group as well
'''

import numpy as np
import numpy.random as rand
from numpy.random import normal,multinomial,binomial
from numpy.random import gamma as gg
import tensorflow as tf
import pickle as pp
from base import SpectralGaussian,Kernels,MatComplex,Mats
from utils import shape,fully_connected_layer,variables_from_scope,toeplitz
from lmc import LMC_DFT,LMC
import os,sys,time
from datetime import datetime as dt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

version = '1.19'

class CSFA_base(object):
	r'''Defines basic methods for encoded CSFA models
	
	Parameters
	----------
	L : int
		Number of components

	reg : float, default=.01
		Regularization for both scores and global parameters

	eta : float, default=5.0
		Fixed noise stength

	Q : int,default=3
		Number of spectral gaussians

	nIter : int,default=1000
		Number of training iterations

	lr_encoder : float,default=1e-4
		The learning rate for our neural network A(X)

	lr_features : float,default=1e-2
		Learning rate for the means and coregionalization matrices

	encoderIter : int,default=25 
		Number of optimization steps for the encoder 

	featureIter : int,default=2
		Number of optimization steps for the features

	R : int, default=2
		Rank of coregionalization matrix

	name: str, default="Default"
		Name of the object. Will be used later when I add saving 
		capabilites
	
	dirName : str, default="./tmp"
		Name of the directory where the model should be saved to. Will be 
		added later
	
	device : int, default=0
		Which GPU to place the jobs on

	percGPU : float, default=.49
		The percent of GPU memory that should be allocated for the object

	trainingMethod : str, default="GradientDescent"
		The gradient descent method to use for optimization. Options are 
		'GradientDescent', 'Momentum', or 'Adam'

	monitorIter : int,default=100
		When I want to see how the parameters change over time this will
		be how often to print them out

	momentum : float, default=.9
		If trainingMethod='Momentum' this is the momentum to use for the
		optimizer
    
	beta1 : float, default=.9
		If trainingMethod='Adam' this is one of the parameters to use 
		for the optimizer

	printIter : int, default=1000
		How often to print out the loss

	batchSize : int, default=100
		For stochastic training the number of data points to include,
		sampled without replacement

	k1 : int, default=256
		Number of nodes in the first layer

	k2 : int, default=64
		Number of nodes in the second layer

	nlayer : int, default=1
		Option for choosing the number of layers (1 or 2)

	activationFunction: str, default='sigmoid'
		The activation function to use, options are 'sigmoid' or 'relu'
	
	init_style : str,default='Uniform'
		Initialization of the means for the spectral Gaussians 

	unif_bounds : array(2,),default=(1,55)
		Bounds on the means with uniform initialization

	batch_size : int,default=100
		The number of datapoints used for gradient descent

	batch_s :
		Number of frequencies to use for stochastic by frequency. Ignored
		for now

	learn_var : bool,default=False
		Learn the variances of the spectral gaussian

	varLam : float,default=1.0
		Put a penatly on the side of the variances per David's suggestion
	
	percGPU : floatin [0,1],default=0.5
		The amount of the GPU to use, can run multiple jobs on single GPU
	
	Attributes
	----------
	creationDate
		Date and time the object was created

	version
		What version the model as being created as

	Methods
	-------
	def getParams(self,session):
		Gets the parameters and takes them out of the graph
	
	def setParams(self,session,params):
		Takes a session and dictionary and sets the graph
	
	def clip(self,session):
		Clips the parameters to be within their bounds

	def bNorm(self):
		Returns the normalization parameter on the coregionalization 
		matricies to ensure identifiability
	
	def _batch(self,Nw):
		Our batching method for the data

	def _definePlaceholdersEncoderLikelihood(self,s,Ns,Nc,Nw):
		Defines all the placeholders and operations to compute the 
		negative log-likelihood

	def _defineOptimization(self):
		Defines the optimizer given specification

	def _defineInitialization(self):
		Just defines the tensorflow init method

	def _defineMonitor(self,monitor):			
		Creates the dictionary needed to monitor parameter learning

	def _updateMonitor(self,sess,myDict,count,monitor):
		Updates the monitor variable with the current variable values

	def _saveGraph(self,saver,session):
		Saves the graph with all the trained parameters
	
	def transform(self,data_real):
		Transforms the data and returns a numpy array
	
	def save_transform(self,data_real,fileName=None):
		Transforms the data and saves it to a file

	Examples
	--------

	References
	----------
	Gallagher, Neil, et al. "Cross-spectral factor analysis." 
			Advances in Neural Information Processing Systems. 2017.
	'''
	def __init__(self,L,reg=.01,eta=5.0,Q=3,nIter=2000,lr_encoder=1e-4,
					lr_features=1e-2,encoderIter=25,featureIter=2,R=2,
					name='Default',dirName='./tmp',device=0,percGPU=0.49,
					trainingMethod='GradientDescent',monitorIter=100,
					momentum=.9,beta1=.9,printIter=1000,k1=256,k2=64,
					nlayer=1,activationFunction='sigmoid',
					init_style='Uniform',unif_bounds=(1,55),
					batch_size=100,learnVar=False,varLam=1.0):
		self.L = int(L) 
		self.Q = int(Q) 
		self.eta = float(eta) 
		self.nIter = int(nIter) 
		self.lr_encoder = float(lr_encoder) 
		self.lr_features = float(lr_features) 
		self.encoderIter = int(encoderIter)
		self.featureIter = int(featureIter)
		self.R = int(R)
		self.reg = float(reg)
		
		self.name = str(name) 
		self.dirName = str(dirName) 
		self.device = int(device) 
		self.version = version 
		
		self.creationDate = dt.now() 
		self.trainingMethod = str(trainingMethod) 
		self.monitorIter = int(monitorIter) 
		self.momentum = float(momentum) 
		self.beta1 = float(beta1) 
		self.printIter = int(printIter) 

		self.k1 = int(k1) 
		self.k2 = int(k2) 
		self.batch_size = int(batch_size) 

		self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=float(percGPU))

		self.learnVar = learnVar 
		self.varLam = float(varLam) 

		self.init_style = init_style
		self.unif_bounds = unif_bounds
		self.activationFunction = str(activationFunction)
		self.nlayer = int(nlayer)

	def getParams(self,session):
		'''get the model parameters out of graph
		Parameters
		----------
		session: tf.Session
			session containing all the variables of the model
		Returns 
		-------
		params: dict
			contains the global parameters as 'LMC'
		'''
		params = {}
		params['LMC'] = [self.LMCkernels[l].getParams(session) for l in range(self.L)]
		return params
	
	def setParams(self,session,params):
		'''
		Parameters
		----------
		session: tf.Session
			session containing all the variables of the model
		params: dict
			Dictionary containing the parameters from getParams
		Returns
		-------
		None
		'''
		if 'LMC' in params:
			for l in range(self.L):
				self.LMCkernels[l].setParams(session,params['LMC'][l])
		if 'encoder' in params:
			pass
	
	def clip(self,session):
		""" Forces the parameters to be within preset bounds
		Parameters
		----------
		session : tf.Session
			Session that we are currently running tensorflow in 
		Returns
		-------
		None
		"""
		for l in range(self.L):
			self.LMCkernels[l].clip(session)

	def bNorm(self):
		'''
		Parameters
		----------
		None
		Returns
		-------
		Bdiagstack : tf.tensor,like=(self.L,self.Q,self.C)
			Gets the diagonals of the coregionalization matrices which we
			use for identifiability due to regularization
		'''
		for l in range(self.L):
			Bdiag = [self.LMCkernels[l].getBdiag() for l in range(self.L)]
		BdiagStack = tf.stack(Bdiag) # L x Q x C
		return BdiagStack
	
	def _batch(self,Nw):
		'''This ramdomly selects data indices. Potentially will be modified
			to cycle through all the data
		Parameters
		----------
		Nw : int
			Number of windows
		Returns
		-------
		idx : bool-like (Nw,)
			Selects batch_size windows used for gradient descent
		'''
		batch = rand.choice(Nw,size=self.batch_size,replace=False)
		idx = np.zeros(Nw)
		idx[batch] = 1
		return (idx==1)

	def _definePlaceholdersEncoderLikelihood(self,s,Ns,Nc,Nw,Np):
		'''
		Parameters
		----------
		s : tensor-like (Ns)
			The frequencies we wish to evaluate the likelihood not used yet
		Ns : int
			Number of frequencies
		Nc : int
			Number of channels
		Nw : int
			Number of windows
		Returns
		-------
		None:
			Creates a bunch of object attributes related to computing NLL
		'''
		########
		# Data #
		########

		# Frequencies are a placeholder in case we decide to do stochastic 
		self.s = tf.placeholder(tf.float32,shape=[Ns],name='s_')
		#The fourier transformed data
		self.y_fft = tf.placeholder(tf.complex64,
					shape=[Ns,Nc,None],name='y_fft')
		#Data used for the encoder. Reals
		self.Y_flat = tf.placeholder(tf.float32,
								shape=[None,Np],name='Y_flat')

		###########
		# Kernels #
		###########
		with tf.variable_scope('global'):
			self.LMCkernels = [LMC(Nc,self.Q,self.R,learnVar=self.learnVar,
					init_style=self.init_style,
					unif_bounds=self.unif_bounds) for i in range(self.L)]

		###########
		# Encoder #
		###########
		with tf.variable_scope('encoder'):
			self.A_enc = tf.Variable(1/self.L*rand.randn(Np,self.L).astype(np.float32),name='A_enc')
			self.B_enc = tf.Variable(1/self.L*rand.randn(self.L).astype(np.float32),name='B_enc')
			self.phi = tf.Variable(1/self.L*rand.randn(3,1).astype(np.float32),name='phi')
			self.bias = tf.Variable(0.1*rand.randn(1).astype(np.float32),
									name='bias')
			self.out_mul = tf.matmul(self.Y_flat,self.A_enc)
			self.out = tf.add(self.out_mul,self.B_enc,name='out')
			self.scores = tf.nn.softplus(self.out,name='scores')
			self.logits = tf.matmul(self.scores[:,:3],self.phi) + self.bias

		###########################
		# Evaluate log-likelihood #
		###########################
		#Combine the factor UKU matrices
		self.UKUL = [self.LMCkernels[l].UKU(self.s) for l in range(self.L)]
		self.UKUstore = tf.stack(self.UKUL)
		self.UKUstorep = tf.transpose(self.UKUstore,perm=[2,3,1,0])

		# Make scores proper dimension
		self.scores_c = tf.cast(tf.transpose(self.scores),
									dtype=tf.complex64)
		self.scores_c1 = tf.expand_dims(self.scores_c,axis=0)
		self.scores_c2 = tf.expand_dims(self.scores_c1,axis=0)
		self.scores_c3 = tf.expand_dims(self.scores_c2,axis=0)
		self.UKUe = tf.expand_dims(self.UKUstorep,axis=-1)

		#Multiply scores
		self.prod_uku = tf.multiply(self.scores_c3,self.UKUe)
		self.prod_ukuT = tf.transpose(self.prod_uku,perm=[4,3,2,0,1])
		self.UKUscores = tf.reduce_sum(self.prod_ukuT,axis=1)

		#Add in the noise 
		self.noise = tf.cast(1/self.eta*tf.eye(self.C),tf.complex64)
		self.UKUnoise_half = tf.add(self.UKUscores,self.noise)
		self.UKUnoise = 2*self.UKUnoise_half

		#Transform Y into the proper shape
		self.Yp = tf.transpose(self.y_fft,perm=[2,0,1])
		self.Yp1 = tf.expand_dims(self.Yp,axis=-1)
		self.Yc = tf.squeeze(tf.conj(self.Yp))

		#Get the quadratic form
		self.SLV = tf.linalg.solve(self.UKUnoise,self.Yp1)
		self.SLVs = tf.squeeze(self.SLV)
		self.Quad = tf.multiply(self.SLVs,self.Yc)
		self.QL = tf.reduce_sum(self.Quad,axis=-1) #Nw x Ns

		#This is where we do the proper weighting 
		self.llk = tf.reduce_mean(self.QL) 

		#Get log determinant
		self.LD = tf.linalg.logdet(self.UKUnoise)

		#Normalization constant
		self.const = tf.cast(Nc*np.log(np.pi)*-1,tf.complex64)

		#Add together for final likelihood
		self.logDet = tf.cast(tf.reduce_mean(self.LD),tf.complex64)
		self.LogLikelihood = self.const - self.logDet - self.llk
		self.eval = tf.real(self.LogLikelihood)

		#################################
		# Final negative log-likelihood #
		#################################
		self.NLL = -1.0*self.eval
		self.logLikelihood = tf.identity(self.NLL,name='LL_')

		# Regularization of scores
		self.reg_scores = 0.01*tf.nn.l2_loss(self.scores)

		#Regularization of factors
		self.reg_features = tf.nn.l2_loss(tf.real(self.bNorm()))

		#################################################
		# Define the final loss with classification etc #
		# somewhere else                                #
		#################################################

		#####################
		# Stuff for UKUnorm #
		#####################
		#self.sf = tf.placeholder(tf.float32,shape=[None],name='sf_')
		s_fine = np.arange(1000)/1000*55
		self.sf = tf.constant(s_fine.astype(np.float32))
		self.UKUL2_norm = [self.LMCkernels[l].UKU(self.sf) for l in range(self.L)]
		self.UKUstore_norm = tf.stack(self.UKUL2_norm)
		self.UKUstorep_norm = tf.transpose(self.UKUstore_norm,perm=[2,3,1,0])
		self.UKUndiv = tf.reduce_sum(tf.abs(self.UKUstorep_norm),axis=3,keepdims=True)
		self.UKUnorm = tf.divide(self.UKUstorep_norm,tf.cast(self.UKUndiv,tf.complex64))

	def _defineOptimization(self):
		'''Defines our optimization algorithm and sets it as a 
		object attribute.
		Parameters
		----------
		None
		Returns
		-------
		None
		'''
		le = self.lr_encoder
		lf = self.lr_features
		if self.trainingMethod == "GradientDescent":
			self.optimstep_l = tf.train.GradientDescentOptimizer(learning_rate=lf).minimize(self.loss,var_list=variables_from_scope('global'))
			self.optimstep_e = tf.train.GradientDescentOptimizer(learning_rate=le).minimize(self.loss,var_list=variables_from_scope('encoder'))
		elif self.trainingMethod == "Momentum":
			self.optimstep_l = tf.train.MomentumOptimizer(learning_rate=lf,
					momentum=self.momentum).minimize(self.loss,
					var_list=variables_from_scope('global'))
			self.optimstep_e = tf.train.MomentumOptimizer(learning_rate=le,
					momentum=self.momentum).minimize(self.loss,
					var_list=variables_from_scope('encoder'))
		elif self.trainingMethod == "Adam":
			self.optimstep_l = tf.train.AdamOptimizer(learning_rate=lf,
					beta1=self.beta1).minimize(self.loss,
					var_list=variables_from_scope('global'))
			self.optimstep_e = tf.train.AdamOptimizer(learning_rate=le,
					beta1=self.beta1).minimize(self.loss,
					var_list=variables_from_scope('encoder'))
			self.optimstep_p = tf.train.AdamOptimizer(learning_rate=le,
					beta1=self.beta1).minimize(self.loss_pred,
					var_list=variables_from_scope('encoder'))
		else:
			print('Unrecognized training method')

	def _defineInitialization(self):
		''' Makes the initialization method and sets it as an attribute
		Parameters
		----------
		None
		Returns
		-------
		None
		'''
		init_global = tf.global_variables_initializer()
		init_local = tf.local_variables_initializer()
		self.init = tf.group(init_global,init_local)
	
	def _saveGraph(self,saver,session):
		'''This saves the tensorflow graph and values to self.dirName
		Paramters
		---------
		saver : tf.Saver
			Should be defined in the fit method
		session : tf.Session
			The session where the model was trained
		Returns
		-------
		None
		'''
		pn = self.dirName + '/' + self.name + '.ckpt'
		save_path = saver.save(session,pn)
		self.chkpt = save_path
		self.meta = self.dirName + '/' + self.name + '.ckpt.meta'
	
	def transform(self,data_real):
		r'''This takes a pretrained model and computes the network 
		stregnths on new data.
		Parameters
		----------
		data_real : np.array-like, (n_samples,n_features)
			The features we use to estimate the scores
		Returns
		-------
		S : np.array-like, (n_samples,n_components)
			Transformed scores of the latent features
		'''
		data_real = data_real.astype(np.float32)
		checkpoint = tf.train.latest_checkpoint(self.dirName)
		with tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options)) as sess:
			new_saver = tf.train.import_meta_graph(self.meta)

			#This restores the graphs along with all the parameter values
			graph = tf.get_default_graph()
			new_saver.restore(sess,checkpoint)

			#Select the two variables we care about
			Y_flat = graph.get_tensor_by_name('Y_flat:0')
			scores = graph.get_tensor_by_name('encoder/scores:0')

			S = sess.run(scores,feed_dict={Y_flat:data_real})

		return S

	def transform_withLikelihood(self,s,data_real,data_complex):
		bs = self.batch_size
		data_real = data_real.astype(np.float32)

		checkpoint = tf.train.latest_checkpoint(self.dirName)
		with tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options)) as sess:
			new_saver = tf.train.import_meta_graph(self.meta)

			#This restores the graphs along with all the parameter values
			graph = tf.get_default_graph()
			new_saver.restore(sess,checkpoint)

			#Select the two variables we care about
			Y_flat_ = graph.get_tensor_by_name('Y_flat:0')
			s_ = graph.get_tensor_by_name('s_:0')
			y_fft_ = graph.get_tensor_by_name('y_fft:0')
			scores = graph.get_tensor_by_name('encoder/scores:0')
			logLikelihood = graph.get_tensor_by_name('LL_:0')

			N = data_real.shape[0]
			S = np.zeros((N,self.L))

			nChunk = int(np.floor(N/self.batch_size))
			nEnd = N-bs*nChunk
			LLs = np.zeros(nChunk)
			for i in range(nChunk):
				print('<<<<<<<<<,',i,nChunk)
				dr_b = data_real[i*bs:(i+1)*bs]
				dc_b = data_complex[:,:,i*bs:(i+1)*bs]
				print('Transform ',i)
				[S_b,LL_b] = sess.run([scores,logLikelihood],
				feed_dict={Y_flat_:dr_b,s_:s,y_fft_:dc_b})
				S[i*bs:(i+1)*bs] = S_b
				LLs[i] = LL_b 

			#Get the last ones at the end
			dr_b = data_real[-bs:]
			dc_b = data_complex[:,:,-bs:]
			[S_b,LL_b] = sess.run([scores,logLikelihood],
			feed_dict={Y_flat_:dr_b,s_:s,
			y_fft_:dc_b})

			S[bs*nChunk:] = S_b[:nEnd]

			LL = (bs*np.sum(LLs)+bs*LL_b)/N

		return S,LL 

	
	def getUKUnorm(self,sess):
		r'''Returns ukunorm
		Paramters
		---------
		sess : tf.Session
			The session where we train the model
		Returns
		-------
		myUKUnorm : np.array-like (n_channels,n_channels,Ns,self.L)
		'''
		myUKUstore_norm = sess.run(self.UKUstore_norm)
		myUKUstorep_norm = sess.run(self.UKUstorep_norm)
		myUKUndiv = sess.run(self.UKUndiv)
		myUKUnorm = sess.run(self.UKUnorm)
		myDict = {'myUKUstore_norm':myUKUstore_norm,
					'myUKUstorep_norm':myUKUstorep_norm,
					'myUKUndiv':myUKUndiv,
					'myUKUnorm':myUKUnorm}
		return myDict

	def transform_inSession(self,data_real,sess):
		r'''This transforms the data without having to load the graph
		Parameters
		----------
		data_real : np.array-like, (n_samples,n_features)
			The features we use to estimate the scores
		sess : tf.Session 
			The session where we train the model
		Returns
		-------
		S : np.array-like, (n_samples,n_components)
			Transformed scores of the latent features
		'''
		S = sess.run(self.scores,feed_dict={self.Y_flat:data_real})
		return S
	
	def save_transform(self,data_real,fileName=None):
		r'''This transforms the data and saves as csv. This saves all 
		networks.
		Parameters 
		----------
		data_real : np.array-like, (n_samples,n_features)
			The features we use to estimate the scores
		fileName : string,optional
			The file we ant to save the model to.
		Returns
		-------
		None
		'''
		S = self.transform(data_real)
		if fileName is None:
			fName = self.name + '_scores.csv'
		else:
			fName = fileName

		np.savetxt(fName,S,fmt='%0.8f',delimiter=',')

	def _defineMonitor(self,monitor):			
		'''This defines our dictionary we use to monitor parameter learning
		Parameters
		----------
		Returns
		-------
		'''

		losses = np.zeros(self.nIter)
		nSave = int(np.ceil(self.nIter/self.monitorIter))
		myDict = {'losses':losses}
		if monitor is None:
			pass
		else:
			if 'means' in monitor:
				#Save spectral gausssian means
				myDict['means'] = np.zeros((nSave,self.L,self.Q))
			if 'vars' in monitor:
				#Save spectral gaussian variances
				myDict['vars'] = np.zeros((nSave,self.L,self.Q))
			if 'logWeights' in monitor:
				#Coregionalization matrices
				myDict['coregs']= np.zeros((nSave,self.L,self.Q,
												self.C,self.R))
			if 'shifts' in monitor:
				#Shift matrices
				myDict['shifts'] = np.zeros((nSave,self.L,self.Q,
												self.C,self.R))

		return myDict

	def _updateMonitor(self,session,myDict,count,monitor):
		'''This automatically updates our tracking
		Parameters
		----------
		Returns
		-------
		'''
		#Added the losses earlier
		params = self.getParams(session)
		LMC = params['LMC']
		for l in range(self.L):
			for q in range(self.Q):
				kernel = LMC[l]['kernels'][q]
				coreg = LMC[l]['coregs'][q]
				if 'means' in monitor:
					#Save spectral gausssian means
					myDict['means'][count,l,q] = kernel['mu']
				if 'vars' in monitor:
					#Save spectral gaussian variances
					myDict['vars'][count,l,q] = kernel['var']
				if 'logWeights' in monitor:
					#Coregionalization matrices
					myDict['coregs'][count,l,q,:,:] = coreg['logWeights']
				if 'shifts' in monitor:
					#Shift matrices
					myDict['shifts'][count,l,q,:,:] = coreg['shifts']
	
	def evaluate(self,s,data_real,data_complex):
		bs = self.batch_size
		Ntimes = int(np.floor(data_real.shape[0]/bs))
		Nw = data_real.shape[0]
		remainder = Nw - Ntimes*bs

		N_tot = 0

		for i in range(Ntimes):
			aa2 = data_complex[:,:,bs*i:(i+1)*bs]
			bb = data_real[bs*i:bs*(i+1)]
			loglike = sess.run(self.NLL,feed_dict={self.s:s,
								self.y_fft:aa2,
								self.Y_flat:bb})
			N_tot += loglike

		return N_tot/Ntimes

class dCSFA_L1adaptive_encoded_dense(CSFA_base):
	def __init__(self,L,reg=.01,eta=5.0,Q=3,nIter=1000,lr_encoder=1e-4,
					lr_features=1e-2,encoderIter=25,featureIter=2,R=2,
					name='Default',dirName='./tmp',device=0,percGPU=0.49,
					trainingMethod='GradientDescent',monitorIter=100,
					momentum=.9,beta1=.9,printIter=1000,
					init_style='Uniform',unif_bounds=(1,55),n_feat=3,
					nlayer=1,activationFunction='sigmoid',phi_init=1.0,
					phi_increase=1.1,phi_decrease=.9,phi_monitor_iter=1,
					batch_size=100,learnVar=False,varLam=1.0,
					mu_max=1.0,mu_start=.01,mu_increase=1.001):
		super(dCSFA_L1adaptive_encoded_dense,self).__init__(L,reg=reg,
				eta=eta,Q=Q,nIter=nIter,R=R,name=name,dirName=dirName,
				device=device,percGPU=percGPU,
				lr_encoder=lr_encoder,lr_features=lr_features,
				encoderIter=encoderIter,featureIter=featureIter,
				trainingMethod=trainingMethod,monitorIter=monitorIter,
				momentum=momentum,beta1=beta1,printIter=printIter,
				init_style=init_style,unif_bounds=unif_bounds,
				batch_size=batch_size,learnVar=learnVar,
				varLam=varLam,nlayer=nlayer,
				activationFunction=activationFunction)

		self.mu_max = float(mu_max)
		self.mu_start = float(mu_start)
		self.mu_increase = float(mu_increase)

		self.init_style = init_style
		self.unif_bounds = unif_bounds
		
		self.phi_init = float(phi_init)
		self.phi_increase = float(phi_increase)
		self.phi_decrease = float(phi_decrease)
		self.phi_monitor_iter = int(phi_monitor_iter)
		self.n_feat = int(n_feat)
	
	def __repr__(self):
		return 'dCSFA_L1_encoded_dense\nL=%d\nQ=%d\eta=%0.3f\nnIter=%d\nLR=%0.8f\nR=%d\nname=%s\ndirName=%s\ndevice=%d\nversion=%s\ntrainingMethod=%s\nk1=%d\nk2=%d\nbatch_size=%d\nn_blessed=%d\nmu=%0.3f'%(self.L,self.Q,self.eta,self.nIter,self.LR,self.R,self.name,self.dirName,self.device,self.version,self.trainingMethod,self.k1,self.k2,self.batch_size,self.n_blessed,self.mu)

	def getParams(self,session):
		params = {}
		params['LMC'] = [self.LMCkernels[l].getParams(session) for l in range(self.L)]
		params['phi'] = session.run(self.phi)
		return params
	
	def setParams(self,session,params):
		if 'LMC' in params:
			for l in range(self.L):
				self.LMCkernels[l].setParams(session,params['LMC'][l])
		if 'encoder' in params:
			pass
			
	
	def _initialize(self,s,Ns,Nc,Nw,Np):
		#Limit ourselves to a particular gpu
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		dev = str(int(self.device))
		os.environ["CUDA_VISIBLE_DEVICES"] = dev

		##########################
		##########################
		###                    ###
		### DEFINING THE GRAPH ###
		###                    ###
		##########################
		##########################
		tf.reset_default_graph()

		#########################
		# Placeholders and data #
		#########################
		self._definePlaceholdersEncoderLikelihood(s,Ns,Nc,Nw,Np)
		self.z_ = tf.placeholder(tf.float32,[self.batch_size])
		self.mu = tf.placeholder(tf.float32,[])
		self.alpha = self.mu/(1. + self.mu)

		#######################
		# Reconstruction Loss #
		#######################
		self.ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.z_,
									logits=tf.squeeze(self.logits))
		self.dloss = tf.reduce_sum(self.ce)

		self.loss_pred = self.dloss + 1/Np*tf.nn.l2_loss(self.A_enc)

		############
		# Sparsity #
		############
		self.phi_loss = tf.reduce_mean(tf.abs(self.phi))

		###########################
		# Final optimization loss #
		###########################
		self.loss = self.alpha*self.NLL + 2*(1-self.alpha)*self.dloss + 0.5*self.reg*self.reg_scores + 0.01*self.reg*self.reg_features + 1*self.phi_loss

		###############################################
		# Define the optimizer based on learning type #
		###############################################
		self._defineOptimization()
			
		########################
		# Initialize the graph #
		########################
		self._defineInitialization()

	def fit(self,s,data_real,data_complex,labels,monitor=None):
		Ns,Nc,Nw = data_complex.shape
		self.C = Nc
		C,Q,R = self.C,self.Q,self.R
		Np = data_real.shape[1]

		####################
		# Create the Graph #
		####################
		self._initialize(s,Ns,Nc,Nw,Np)

		############################################################
		# Here we create the variables for monitoring the training #
		############################################################
		nSave = int(np.ceil(self.nIter/self.monitorIter))
		training = self._defineMonitor(monitor)
		training['dlosses'] = np.zeros(self.nIter)
		training['NLL'] = np.zeros(self.nIter)
		training['pr'] = np.zeros(self.nIter)

		if monitor is None:
			pass
		else:
			if 'phis' in monitor:
				myPhi = np.zeros((nSave,self.L))

		count = 0
		startTime = time.time()
		sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))

		mu = self.mu_start

		saver = tf.train.Saver()

		if 1 == 1:
			sess.run(self.init)

			for i in range(self.nIter):
				print('>>',i)
				mu = np.minimum(mu*self.mu_increase,self.mu_max)
				idx = self._batch(Nw)
				alpha = mu/(1. + mu)

				bb = data_real[idx]
				aa2 = data_complex[:,:,idx]
				zz = labels[idx]

				for j in range(self.featureIter):
					try:
						sess.run(self.optimstep_l,feed_dict={self.s:s,
								self.mu:mu,
								self.y_fft:aa2,
								self.Y_flat:bb,
								self.z_:zz})
					except:
						print('1')
						training['scores'] = sess.run(self.scores,
									feed_dict={self.s:s,self.y_fft:aa2,
									self.mu:mu,
									self.Y_flat:bb,
									self.z_:zz})
						self._updateMonitor(sess,training,count,monitor)
						params = self.getParams(sess)
						training['idxs'] = idx
						return params,training,sess

				for iii in range(self.encoderIter):
					try:
						_,ll,dl,loglike = sess.run([self.optimstep_e,
								self.loss,self.dloss,self.NLL],
								feed_dict={self.s:s,
								self.mu:mu,
								self.y_fft:aa2,
								self.Y_flat:bb,
								self.z_:zz})
					except:
						print('2')
						training['scores'] = sess.run(self.scores,
										feed_dict={self.s:s,self.y_fft:aa2,
										self.mu:mu,
										self.Y_flat:bb,
										self.z_:zz})
						self._updateMonitor(sess,training,count,monitor)
						params = self.getParams(sess)
						training['idxs'] = idx
						return params,training,sess

				for jj in range(20):
					_  = sess.run(self.optimstep_p,
								feed_dict={self.s:s,
								self.mu:mu,
								self.y_fft:aa2,
								self.Y_flat:bb,
								self.z_:zz})

				#########################################################
				# Here is where we monitor the variables we've selected #
				#########################################################
				training['dlosses'][i] = dl
				training['losses'][i] = ll
				training['NLL'][i] = loglike
				#if i%self.monitorIter==0:
				#	self._updateMonitor(sess,training,count,monitor)
				#	if 'phis' in monitor:
				#		myPhi[count,:] = np.squeeze(sess.run(self.phi))
				#	count = count + 1

				#This is our adaptive sparsity

				#This prints the training progress
				if i%self.printIter == 0:
					el = time.time() - startTime
					sl = self.reg*sess.run(self.reg_scores,
											feed_dict={self.s:s,
											self.mu:mu,
											self.y_fft:aa2,
											self.Y_flat:bb,
											self.z_:zz})
					fl = self.reg*sess.run(self.reg_features,
											feed_dict={self.s:s,
											self.mu:mu,
											self.y_fft:aa2,
											self.Y_flat:bb,
											self.z_:zz})
					nl = self.reg*sess.run(self.NLL,
											feed_dict={self.s:s,
											self.mu:mu,
											self.y_fft:aa2,
											self.Y_flat:bb,
											self.z_:zz})
					print('Iteration %d,Time = %0.1f,Loss = %0.3f,dLoss=%0.3f,scores=%0.3f,feature=%0.3f,NLL=%0.3f'%(int(i),el,ll,dl,sl,fl,nl))
					print(sess.run(self.phi))

			self._saveGraph(saver,sess)

		if 'phis' in monitor:
			training['phis'] = myPhi
		params = self.getParams(sess) 
		return params,training,sess
	

def main():
	tf.reset_default_graph()
	#Load data
	#data = some complex data
	s = np.arange(1,120)
	N,C,W = data.shape

	model = CSFA_gd(N,C,W)
	model.fit_transform(s,data)

if __name__ == "__main__":
	main()


