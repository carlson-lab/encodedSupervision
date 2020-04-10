''' 
This contains our basic global parameters for CSFA, the spectral gaussian
kernel and coregionalization matrix. 
Version history:
Version 1.0, 1/1/19:
	Started this
Version 1.1, 3/4/19:
	Added significant documentation as well as cleaning up the code
'''
import numpy as np
import numpy.random as rand
import tensorflow as tf
from numpy.random import gamma,normal,multinomial,binomial
import pickle as pp
from utils import shape,fully_connected_layer,variables_from_scope,toeplitz
from utils import size

#####################################################################
# Defining global constants on the bounds on the spectral means and #
# variances. It also defines pi and pi^2                            #
#####################################################################
mymuB= np.zeros(2).astype(np.float32)
mymuB[0] = 1
mymuB[1] = 120
myvB= np.zeros(2).astype(np.float32)
myvB[0] = np.exp(-1.3863)
myvB[1] = np.exp(6.4378)
pi = np.pi
pi2 = np.pi*np.pi

class SpectralGaussian(object):
	'''Spectral gaussian kernel
	This object has the mean and variance as tensorflow variables. 
	The two mai methods are getting the covariance via covFun and the 
	spectral density via specDens. scpecDens and covFun return 
	*TENSORS* which we need in order to have automatic differentiation.

	Parameters
	----------
	mu : tf.Variable, shape = []
		The spectral mean. Can be initialized as uniform or gamma
	Methods
	-------
	def __repr__(self,session)
		Method to print the object parameters (requires session)
	def clip(self,session)
		Forces variables to be within bounds (done outside graph)
	def getParams(self,session)
		Gets all relevant paramters (outside of graph)
	def getMu(self,session)
		Gets mean (outside of graph)
	def getVar(self,session)
		Gets variance (outside of graph)
	def getVarB(self,session)
		Gets variance bound (outside of graph)
	def getMuB(self,session)
		Gets mean bounds (outside of graph)
	def setMu(self,session,mu)
		Sets the mean value (outside of graph)
	def setVar(self,session,var)
		Sets the var value (outside of graph)
	def setParams(self,session,params)
		Sets the params (outside of graph)
	def covFun(self,tau)
		Gets covariance based on tau (in graph, returns tensor)
	def specDens(self,s)
		Gets spectral density (in graph, returns tensor)
	def save(self,fileName)
		Saves model as pickle object
	Attributes
	----------
	self.mu_assign 
	self.var_assign 
	self.assign_mu
	self.clip_mu
	self.assign_var
	self.clip_var
	self.clip_vars
		All of these are used for clipping and setting

	Examples
	--------
	import numpy as np
	import tensorflow as tf
	import numpy.random as rand
	from numpy.random import gamma,normal,multinomial,binomial
	import base

	tf.reset_default_graph()
	#Var as constrant 
	model = base.SpectralGaussian()
	#Var as variable
	model_var = base.SpectralGaussian(mu=15.5,var=3,learnVar=True)
	sess = tf.Session() 
	tf.global_variables_initializer().run(session = sess)

	print('Test print')
	print('Fixed variance')
	model.__repr__(sess)
	print('Learned variance')
	model_var.__repr__(sess)

	print('Testing our getters')
	print('Fixed variance')
	print(model.getMu(sess))
	print(model.getVar(sess))
	print(model.getVarB(sess))
	print(model.getMuB(sess))
	print(model.getParams(sess))
	print('Learned variance')
	print(model_var.getMu(sess))
	print(model_var.getVar(sess))
	print(model_var.getVarB(sess))
	print(model_var.getMuB(sess))
	print(model_var.getParams(sess))

	References
	----------
	Gallagher, Neil, et al. "Cross-spectral factor analysis." Advances 
		in Neural Information Processing Systems. 2017.
	'''
	def __init__(self,mu=None,var=None,muB=None,varB=None,learnVar=False,
						init_style='Uniform',unif_bounds=(1,55)):
		#Initialize the mean
		if mu is None:
			if init_style is 'Uniform':
				myMu = rand.uniform(low=unif_bounds[0],high=unif_bounds[1])
				self.mu =  tf.Variable(myMu,dtype=tf.float32)
			elif init_style is 'Gamma':
				self.mu =  tf.Variable(2+gamma(10,1),dtype=tf.float32)
			else:
				print('Unrecognized initialization')
		else:
			self.mu = tf.Variable(mu,dtype=tf.float32)

		#Initialize the variance
		if learnVar:
			if var is None:
				self.var = tf.Variable(1+gamma(2,0.5),dtype=tf.float32)
			else:
				self.var = tf.Variable(var,dtype=tf.float32)
		else:
			if var is None:
				setVar = np.log(5)
			else:
				setVar = var
			self.var = tf.constant(setVar,dtype=tf.float32)

		#Bounds on the mean
		if muB is None:
			self.muB = tf.constant(mymuB,dtype=tf.float32)
		else:
			self.muB = tf.constant(muB,dtype=tf.float32)

		#Bounds on the variance
		if varB is None:
			self.varB = tf.constant(myvB,dtype=tf.float32)
		else:
			self.varB = tf.constant(varB,dtype=tf.float32)

		#Number of parameters
		self.nParams = int(2)
		self.learnVar = learnVar

		#Setting operations
		self.mu_assign = tf.placeholder('float',[])
		self.var_assign = tf.placeholder('float',[])
		self.assign_mu = tf.assign(self.mu,self.mu_assign)

		#These are used to bound the parameters
		self.clip_mu = tf.assign(self.mu,
						tf.clip_by_value(self.mu,self.muB[0],self.muB[1]))

		if learnVar:
			self.assign_var = tf.assign(self.var,self.var_assign)
			self.clip_var = tf.assign(self.var,
					tf.clip_by_value(self.var,self.varB[0],self.varB[1]))
			self.clip_vars = tf.group(self.clip_mu,self.clip_var)
	
	#This prints out a representation of the object within the session
	def __repr__(self,session):
		mu = self.getMu(session)
		var = self.getVar(session)
		muB = self.getMuB(session)
		varB = self.getVarB(session)
		return 'SpectralGaussian\nMu=%0.3f\nVar=%0.3f\nMuB=[%0.3f,%0.3f]\nVarB=[%0.3f,%0.3f]\n'%(mu,var,muB[0],muB[1],varB[0],varB[1])
	
	#This clips the values in the object
	def clip(self,session):
		session.run(self.clip_vars)
	
	#These are the getters. Values are always returned as floats so it 
	#can be used in other python programs/debugging
	def getParams(self,session):
		params = {'mu':self.getMu(session),'var':self.getVar(session)}
		return params

	def getMu(self,session):
		return session.run(self.mu)

	def getVar(self,session):
		return session.run(self.var)

	def getVarB(self,session):
		return session.run(self.varB)

	def getMuB(self,session):
		return session.run(self.muB)
	
	#These are the setters. Don't need these for training but manually 
	#changing the values
	def setMu(self,session,mu):
		session.run(self.assign_mu,feed_dict={self.mu_assign:mu})

	def setVar(self,session,var):
		session.run(self.assign_var,feed_dict={self.var_assign:var})

	def setParams(self,session,params):
		session.run(self.assign_mu,feed_dict={self.mu_assign:params['mu']})
		if self.learnVar:
			session.run(self.assign_var,
					feed_dict={self.var_assign:params['var']})

	def covFun(self,tau):
		'''
		Evaluates the covariance function given as equation 5
		Inputs
			tau : can be a float, placeholder, or array of single dimension
		Output
			ksg : complexTensor, (len(tau),)
		'''
		real1 = -2.*pi2*tau*tau*self.var
		imag1 = 2*pi*tau*self.mu
		comp1 = tf.complex(real1,imag1)
		return tf.exp(comp1)

	def specDens(self,s):
		'''
		Evaluates the spectral density as given in equation 4
		Inputs
			s : float, placeholder, or numpy array of single dimension
		Output
			Ssg : real, (len(s),)
		'''
		diff = self.mu-s
		var = tf.exp(self.var)
		#return 1/tf.sqrt(2.0*pi*self.var)*tf.exp(-.5/self.var*diff*diff)
		return 1/tf.sqrt(2.0*pi*var)*tf.exp(-.5/var*diff*diff)
	
	def save(self,fileName):
		myDict = {'model':self}
		pp.dump(myDict,open(fileName,'wb'))

'''
Kernels
This object has a list of spectral gaussian kernels. Once again there 
are methods that access the actual variables as floats within a session 
but the two key methods of evaluating the covariance and spectral density
return tensors.
_______________
Attributes
  Q		- number of kernels
  k		- list of kernels
  nParams - number of parameters
_______________
Methods
getParams 	- gets the means and variances as a numpy matrix
covFun		- evaluate the cov at \tau. Can be an array. Returns tensor. 
			  Can be for a specific kernel or all of them as an array
specDens	- evaluate the density at s. Can be an array. Returns tensor
			  Can be for a specific kernel or all of them as an array
'''
class Kernels(object):
	def __init__(self,init_style='Uniform',unif_bounds=(1,55),
						mu_bounds=None,Q=None,k=None,learnVar=False):
		#Default number of gaussians is 3
		if Q is None:
			self.Q = int(3)
		else:
			self.Q = int(Q)

		#If no list of kernels is provided make a list of kernels using
		#default initialization
		if k is None:
			self.k = [SpectralGaussian(init_style=init_style,
						unif_bounds=unif_bounds,
						learnVar=learnVar) for i in range(self.Q)]
		else:
			self.k = k

		self.nParams = int(self.Q*2)
	
	def __repr__(self):
		return 'Kernels\nQ=%d'%(self.Q)
	
	def clip(self,session):
		for q in range(self.Q):
			self.k[q].clip(session)
	
	def vars(self):
		vlist = [self.k[i].var for i in range(self.Q)]
		vval = tf.stack(vlist)
		vs = tf.exp(vval)
		return vs

	
	#This extracts all the parameters of interest in the spectral gaussians
	#List of dictionaries
	def getParams(self,session,q=None):
		if q is None:
			params = [self.k[i].getParams(session) for i in range(self.Q)]
		else:
			params = self.k[q].getParams(session)
		return params
	
	#This sets the parameters
	def setParams(self,session,params):
		for i in range(self.Q):
			self.k[i].setParams(session,params[i])
	
	def covFun(self,tau,q=None):
		'''
		Evaluates the covariance function given as equation 5
		Inputs
		tau : can be a float, placeholder, or array of single dimension
		Output
		ksg : complexTensor, (len(tau),Q)
		'''
		if q is None:
			cf = tf.stack([self.k[i].covFun(tau) for i in range(self.Q)])
		else:
			cf = self.k[q].covFun(tau)
		return cf
			
	def specDens(self,s,q=None):
		'''
		Evaluates the spectral density as given in equation 4
		Inputs
		s : can be a float, placeholder, or numpy array of single dimension
		Output
		Ssg : real, (len(s),Q)
		'''
		if q is None:
			sf = tf.stack([self.k[i].specDens(s) for i in range(self.Q)])
		else:
			sf = self.k[q].specDens(s)
			
		return sf

	def save(self,fileName):
		myDict = {'model':self}
		pp.dump(myDict,open(fileName,'wb'))

'''
MatComplex
This object stores the coregionalization matrix which we have as a reduced 
rank R
'''
class MatComplex(object):
	def __init__(self,C,R,weights=None,shifts=None,
											weightsB=None,shiftsB=None):
		LOGW_LB = -9.9
		LOGW_UB = 3
		S_LB = -1000000
		S_UB = 1000000
		self.C = int(C)
		self.R = int(R)
		if weights is None:
			lw_init = np.log(.1*rand.random((C,R))+np.exp(LOGW_LB))
			self.logWeights = tf.Variable(lw_init,dtype='float')
		else:
			self.logWeights = tf.Variable(np.log(weights),dtype='float')
		if shifts is None:
			s_init = pi/2*normal(size=(C,R))
			s_init[0] = 0
			self.shifts = tf.Variable(s_init,dtype='float')
		else:
			self.shifts = tf.Variable(shifts,dtype='float')
		if weightsB is None:
			wb = np.ones((C,R,2))
			wb[:,:,0] = LOGW_LB
			wb[:,:,1] = LOGW_UB
			self.weightsB = tf.constant(wb,dtype='float')
		else:
			self.weightsB = tf.constant(weightsB,dtype='float')
		if shiftsB is None:
			sb = np.ones((C,R,2))
			sb[:,:,0] = S_LB
			sb[:,:,1] = S_UB
			self.shiftsB = tf.constant(wb,dtype='float')
		else:
			self.shiftsB = tf.constant(shiftsB,dtype='float')

		self.nParams = int(C*R+(C-1)*R)
		self.iMat = tf.constant(1j*np.ones((C,R)),dtype=tf.complex64)

		#Setting operations
		self.w_assign = tf.placeholder('float',[C,R])
		self.s_assign = tf.placeholder(dtype='float',shape=[C,R])
		self.assign_weights = tf.assign(self.logWeights,self.w_assign)
		self.assign_shifts = tf.assign(self.shifts,self.s_assign)

		#These are used to bound the parameters
		self.clip_weights = tf.assign(self.logWeights,
				tf.clip_by_value(self.logWeights,self.weightsB[:,:,0],
				self.weightsB[:,:,1]))
		self.clip_shifts = tf.assign(self.shifts,
				tf.clip_by_value(self.shifts,self.shiftsB[:,:,0],
				self.shiftsB[:,:,1]))
		self.clip_vars = tf.group(self.clip_weights,self.clip_shifts)
	
	def __repr__(self):
		return 'MatComplex\nC=%d\nR=%d\nnParams=%d'%(self.C,self.R,
														self.nParams)
	
	#This clips the values in the object
	def clip(self,session):
		session.run(self.clip_vars)

	def getMat(self):
		w = tf.exp(self.logWeights)
		sc = tf.cast(self.shifts,tf.complex64)
		wc = tf.cast(w,tf.complex64)
		compMul = tf.multiply(self.iMat,sc)
		ecm = tf.exp(compMul)
		b = tf.multiply(wc,ecm)
		bc = tf.conj(b)
		bh = tf.transpose(bc)
		return tf.transpose(tf.matmul(b,bh))
	
	def getParams(self,session):
		lw = session.run(self.logWeights)
		shifts = session.run(self.shifts)
		params = {'logWeights':lw,'shifts':shifts}
		return params

	#These are the setters. Don't need these for training but manually 
	#changing the values
	def setWeights(self,session,weights):
		session.run(self.assign_weights,feed_dict={self.w_assign:weights})

	def setShifts(self,session,shifts):
		session.run(self.assign_shifts,feed_dict={self.s_assign:shifts})

	def setParams(self,session,params):
		session.run(self.assign_weights,
						feed_dict={self.w_assign:params['logWeights']})
		session.run(self.assign_shifts,
						feed_dict={self.s_assign:params['shifts']})

	def save(self,fileName):
		myDict = {'model':self}
		pp.dump(myDict,open(fileName,'wb'))

'''
Mats
This object has a list of coregionalization matrices. The same methds as 
previously
_______________
Attributes
  Q		- number of kernels
  C		- number of channels
  B		- list of coregionalization matrices
  nParams - number of parameters
_______________
Methods
'''
class Mats(object):
	def __init__(self,Q,C,R,B=None):
		self.Q = int(Q)
		self.C = int(C)
		if isinstance(R,float):
			myR = R*np.ones(self.Q)
		elif isinstance(R,int):
			myR = R*np.ones(self.Q)
		else:
			myR = R

		if B is None:
			self.B = [MatComplex(C,int(myR[i])) for i in range(self.Q)]
		else:
			self.B = B

		self.nParams = self.Q*self.B[0].nParams
	
	def __repr__(self,session):
		return 'Mats\nQ=%d\nC=%d'%(self.Q,self.C)
	
	#This just clips everything
	def clip(self,session):
		for q in range(self.Q):
			self.B[q].clip(session)

	#Return list of dicts
	def getParams(self,session):
		params = [self.B[i].getParams(session) for i in range(self.Q)]
		return params

	def setParams(self,session,params):
		for i in range(self.Q):
			self.B[i].setParams(session,params[i])

	def getMat(self,q):
		return self.B[q].getMat()
	
	def getMats(self):
		return [self.B[i].getMat() for i in range(self.Q)]
	
	def getMatsVec(self):
		pass


def main():
	pass

if __name__ == "__main__":
	main()





