import numpy as np
import tensorflow as tf
import numpy.random as rand

def shape(tensor):
    """
    Get the shape of a tensor. This is a compile-time operation,
    meaning that it runs when building the graph, not running it.
    This means that it cannot know the shape of any placeholders
    or variables with shape determined by feed_dict.
    """
    return tuple([d.value for d in tensor.get_shape()])

def size(tensor):
	tt = tuple([d.value for d in tensor.get_shape()])
	L = len(tt)
	prod = 1
	for l in range(L):
		prod = prod*tt[l]
	return prod

def fully_connected_layer(in_tensor, out_units, activation_function=tf.nn.relu):
    """
    Add a fully connected layer to the default graph, taking as input `in_tensor`, and
    creating a hidden layer of `out_units` neurons. This should be called within a unique variable
    scope. Creates variables W and b, and computes activation_function(in * W + b).
    """
    _, num_features = shape(in_tensor)
    W = tf.get_variable("weights", [num_features, out_units], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable("biases", [out_units], initializer=tf.constant_initializer(0.1))
    return activation_function(tf.matmul(in_tensor, W) + b)

def variables_from_scope(scope_name):
    """
    Returns a list of all variables in a given scope. This is useful when
    you'd like to back-propagate only to weights in one part of the network
    (in our case, the generator or the discriminator).
    """
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)

def getSymm(inMat):
    LT = tf.matrix_band_part(inMat,-1,0)
    LT2 = inMat - tf.matrix_band_part(inMat,0,-1)
    aj = tf.conj(LT2)
    LTT = tf.transpose(LT)
    out = LTT + aj
    return out

def toeplitz(inVec):
    c = tf.fft(tf.cast(inVec,tf.complex64))
    operator = tf.linalg.LinearOperatorCirculant(c)
    out = operator.to_dense()
    symm = getSymm(out)
    return symm
