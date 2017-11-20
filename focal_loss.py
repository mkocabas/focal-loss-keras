from keras import backend as K

def focal_loss(gamma=2, alpha=2):
	def focal_loss_fixed(y_true, y_pred):
		if(K.backend()=="tensorflow"):
			import tensorflow as tf
			pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
			return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt))
		if(K.backend()=="theano"):
			import theano.tensor as T
			pt = T.where(T.eq(y_true, 1), y_pred, 1 - y_pred)
			return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt))
	return focal_loss_fixed
