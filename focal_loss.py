from keras import backend as K
import tensorflow as tf

# Compatible with tensorflow backend

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -alpha * K.mean(K.pow(1. - pt_1, gamma) * K.log(pt_1)) - (1 - alpha) * K.mean(K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed
