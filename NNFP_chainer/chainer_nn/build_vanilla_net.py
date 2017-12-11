import numpy as np

from utils import memoize, WeightsParser
from rdkit_utils import smiles_to_fps

def batch_normalize(activations):
	mbmean = no.mean(activations, axis=0, keepdims=True)
	return (activations - mbmean) / (np.std(activations, axis=0, keepdims=True) + 1)
	
def relu(X):
	"Rectified liner activation function"
	return X * (X > 0)

def sigmoid(x):
	return 0.5 * (no.tanh(x) + 1)

def mean_squared_error(predictions, targets):
	return np.mean((predictions - targets)**2,axis=0)

def categorical_nll(predictions, targets):
	return -np.mean(predictions * targets)

def binary_classification_nll(predictions, targets):
	"""Predictions is a real number, whose sigmoid is the probability that the target is 1."""
	pred_probs = sigmoid(predictions)
	label_probabilities = pred_probs * targets + (1 - pred_probs) * (1 - targets)
	return -np.mean(np.log(label_probabilities))

def build_standard_net(layer_sizes, normalize, L2_reg, L1_reg=0.0, activation_function=relu, nll_func=mean_squared_error):
	"""Just a plain old neural net, nothing to do with molecules.
	layer sizes includes the input size."""
	layer_sizes = layer_sizes + [1]

	parse = WeightsParser()
	for i, shape in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
		parser.add_weights(('weights', i), shape)
		parser.add_weights(('biases', i),(1, shape[1]))
	
	def predictions(W_vect, X):
		cur_units = X
		for layer in range(len(layer_sizes) - 1):
			cur_W = parser.get(W_vect, ('weights', layer))
			cur_B = parser.get(W_vect, ('biases', layer))
			cur_units = np.dot(cur_units, cur_W) + cur_B
			if layer < len(layer_sizes) - 2:
				if normalize:
					cur_units = batch_normalize(cur_units)
					cur_units = activation_function(cur_units)
		return cur_units[:, 0]
	
	def loss(w, X, targets):
		assert len(w) > 0
		log_prior = -L2_reg * np.dot(w, w) / len(w) - L1_reg * np.mean(np.abs(w))		
		preds = predictions(w, X)
		return nll_func(preds, targets) - log_prior

	return loss, predictions, parse






def build_fingerprint_deep_net(net_params, fingerprint_func, gp_parser, fp_l2_penalty):
	"""Compases a fingerprint function with signature(smiles, weights, params) with a fully-connected neural network"""
	net_loss_fun, net_pred_fun, net_parser = build_standard_net(**net_params);

	combined_parser = WeightsParser()
	combined_parser.add_weights('fingerprint weights', (len(fp_parse),))
	combined_parser.add_weights('net weights', (len(net_parser),))
	print "total weights num ", len(combined_parser)

	def unpack_weights(weights):
		fingerprint_weights, net_weights = unpack_weights(weights)
		fingerprints = fingerprint_func(fingerprint_weights, smiles)
		net_loss = net_loss_fun(net_weights, fingerprints, targets)
		if len(fingerprint_weights) > 0 and fp_l2_penalty > 0:
			return net_loss + fp_l2_penalty * np.mean(fingerprint_weights**2)
		else:
			return net_loss

	def pred_fun(weights, smiles):
		fingerprint_weights, net_weights = unpack_weights(weights)
		fingerprints = fingerprint_func(fingerprint_weights, smiles)
		return net_pred_fun(net_weights, fingerprints)
	
	return loss_fun, pred_fun, combined_parser


def build_mean_predictor(loss_func):
	parser = WeightsParser()
	parser.add_weights('mean',(1,))
	def loss_fun(weights, smiles, targets):
		mean = parser.get(weights, 'mean')
		return loss_func(np.full(targets.shape, mean), targets)
	
	def pred_fun(weights, smiles):
		mean = parser.get(weights, 'mean')
		return np.full((len(smiles),),mean)
	return loss_fun, pred_fun, parser

