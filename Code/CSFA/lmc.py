'''
This contains LMC and LMC_DFT
'''
import tensorflow as tf
import numpy as np
import numpy.random as rand
from numpy.random import normal,multinomial,binomial
from numpy.random import gamma as gg
from base import SpectralGaussian,Kernels,MatComplex,Mats
from utils import shape,fully_connected_layer,variables_from_scope,toeplitz
from utils import size

class LMC(object):
	def __init__(self,C,Q,R=2,coregs=None,kernels=None,gamma=None,
						init_style='Uniform',unif_bounds=(1,55),
						learnVar=False):
		self.C = int(C)
		self.Q = int(Q)
		self.R = int(R)

		#Here are the coregionalization matrices
		if coregs is None:
			self.coregs = Mats(self.Q,self.C,R)
		else:
			self.coregs = coregs

		#Here are the kernels
		if kernels is None:
			self.kernels = Kernels(init_style=init_style,
						unif_bounds=unif_bounds,Q=self.Q,learnVar=learnVar)
		else:
			self.kernels = kernels

		if gamma is None:
			self.gamma = tf.constant(100000.,dtype='float')
		else:
			self.gamma = tf.constant(float(gamma),dtype='float')

		self.nParams = self.kernels.nParams + self.coregs.nParams + 1

	def __repr__(self):
		return 'LMC\nC=%d\nQ=%d\nR=%d'%(self.C,self.Q,self.R)
	
	def clip(self,session):
		self.kernels.clip(session)
		self.coregs.clip(session)
	
	#We are going to return these as a dictionary
	def getParams(self,session):
		params = {}
		params['kernels'] = self.kernels.getParams(session)
		params['coregs'] = self.coregs.getParams(session)
		params['gamma'] = session.run(self.gamma)
		return params

	#Params is a dictionary
	def setParams(self,session,params):
		if params['kernels'] is not None:
			self.kernels.setParams(session,params['kernels'])
		if params['coregs'] is not None:
			self.coregs.setParams(session,params['coregs'])
	
	def vars(self):
		return self.kernels.vars()

	#Returns a tensor
	#NEVER USED IN CSFA
	def logEvidence(self,tau,y):
		K = tf.real(self.K(tau))
		logdetK = tf.linalg.logdet(K)
		N = tf.size(tau)
		W = shape(y)[2]
		#y = reshape(y,[N*self.C,W])
		y2 = tf.reshape(y,[N*self.C,W])

		res = -0.5*N*self.C*tf.log(2*pi)-0.5*logdetK
		res2 = res - 0.5*tf.reduce_sum(tf.multiply(y2,tf.solve(K,y2),1))
		return res2

	#Returns a tensor
	#NEVER USED IN CSFA
	def evaluate(self,tau,y):
		return tf.reduce_mean(self.logEvidence(tau,y))
	
	def K(self,tau):
		N = shape(tau)[0]
		Bs = [self.coregs.getMat(q) for q in range(self.Q)]#List of tensors
		BLO = [tf.linalg.LinearOperatorFullMatrix(b) for b in Bs]
		cv = model.kernels.covFun(tau)
		Ks = [toeplitz(cv[i]) for i in range(shape(cv)[0])]
		KSO = [tf.linalg.LinearOperatorFullMatrix(k) for k in Ks]
		BKs = [tf.linalg.LinearOperatorKronecker([BLO[i],KSO[i]]) for i in range(self.Q)]
		BKS = [mat.to_dense() for mat in BKs]
		resNoNoise = tf.add_n(BKS)
		resNoise = resNoNoise + tf.cast(tf.scalar_mul(1./self.gamma,tf.eye(N*self.C)),tf.complex64)
		return resNoNoise,resNoise
	
	def getBdiag(self):
		Bs = self.coregs.getMats()
		BBT = [tf.matmul(Bs[i],tf.transpose(Bs[i])) for i in range(self.Q)]
		#BBtd = [tf.linalg.tensor_diag_part(BBT[i]) for i in range(self.Q)]
		BBtd = [tf.diag_part(BBT[i]) for i in range(self.Q)]
		BdiagStack = tf.stack(BBtd) #Q x C
		return BdiagStack

	
	def covFuns(self,tau):
		N = tf.size(tau)
		Bs = self.coregs.getMats()#Returns list
		dBs = [tf.diag_part(Bs[i]) for i in range(len(Bs))]#Diagonals
		dbE = [tf.expand_dims(dBs[i],axis=0) for i in range(len(dBs))]#Expands on first dim
		ks = [tf.transpose(tf.real(self.kernels.covFun(tau,q))) for q in range(self.Q)]
		dbEr = [tf.cast(dbE[i],dtype=tf.float32) for i in range(len(dbE))]
		res_list = [tf.multiply(ks[i],dbEr[i]) for i in range(len(ks))]
		res = tf.add_n(res_list) + 1./self.gamma
		return res
	
	def normalizeCovariance(self,session):
		maxCov = session.run(tf.reduce_max(self.covFuns(0)))
		ps = self.coregs.getParams(session)
		for q in range(self.Q):
			ps[q]['logWeights'] = ps[q]['logWeights']-0.5*np.log(maxCov)
		self.coregs.setParams(session,ps)

		#self.gamma = self.gamma*maxCov
		return maxCov
	
	#Return 3d tensor
	#Does not add noise
	def UKU(self,s):
		ss = tf.size(s)
		Ns = tf.to_float(tf.size(s))
		d = tf.cast(0.5/Ns/(s[2]-s[1]),dtype=tf.complex64)
		Nc = self.C
		SD = self.kernels.specDens(s)
		B = self.coregs.getMats()
		Bcomb = tf.stack(B)
		BC = tf.expand_dims(Bcomb,axis=1)
		SD1 = tf.expand_dims(SD,axis=-1)
		SD2 = tf.cast(tf.expand_dims(SD1,axis=-1),dtype=tf.complex64)
		myMult = tf.multiply(BC,SD2)
		UKU = tf.reduce_sum(myMult,axis=0)
		return 0.5*UKU/d

class LMC_DFT(LMC):
	def __init__(self,*args,**kwargs):
		super(LMC_DFT,self).__init(*args,**kwargs)

	def __repr__(self):
		pass

	#NEVER USED IN CSFA
	def evaluate(self,s,yfft,UKUinv=None):
		if UKUinv is None:
			UKUinv = self.getUKUinv(s,opts)

		res = tf.reduce_sum(self.logEvidence(yfft,UKUinv))
		return res 

	#NEVER USED IN CSFA
	def logEvidence(self,yfft,UKUinv):
		pass
		#Ns = tf.size(yfft)
		#UKUinv2 = tf.scalar_mul(0.5,UKUinv)

	#Only used in gradient or evaluate, neither of which are needed
	def getUKUinv(self,s,opts):
		pass


		

def main():
	pass

if __name__ == "__main__":
	main()
