#coding: utf-8
import numpy as np
import numpy.random as npr
#import cupy as cp #GPUを使うためのnumpy
import chainer 
from chainer import cuda, Function, gradient_check, \
	Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import random
from collections import OrderedDict
from features import num_atom_features, num_bond_features
from mol_graph import degrees
from build_convnet import matmult_neighbors, array_rep_from_smiles, softmax, sum_and_stack
from build_vanilla_net import batch_normalize


class WeightsParser(object):
    """A kind of dictionary of weights shapes,
       which can pick out named subsets from a long vector.
       Does not actually store any weights itself."""
    def __init__(self):
        self.idxs_and_shapes = OrderedDict()
        self.N = 0

    def get(self, vect, name):
        """Takes in a vector and returns the subset indexed by name."""
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)

    def set(self, vect, name, value):
        """Takes in a vector and returns the subset indexed by name."""
        idxs, _ = self.idxs_and_shapes[name]
        vect[idxs] = np.ravel(value)

    def __len__(self):
        return self.N


def weights_name(layer, degree):
    return "layer_" + str(layer) + "_degree_" + str(degree) + "_filter"

def bool_to_float32(features):
	return np.array(features).astype(np.float32)	

class FP(Chain):
	def __init__(self, model_params):
		super(FP, self).__init__()
		with self.init_scope():
			setattr(self, 'model_params', model_params)
			parser = WeightsParser()
			num_hidden_features = [self.model_params['conv_width']] * self.model_params['fp_depth']
			all_layer_sizes = [num_atom_features()] + num_hidden_features

			for layer in range(len(all_layer_sizes)):
				setattr(self, 'layer_output_weights_'+str(layer), L.Linear(all_layer_sizes[layer], self.model_params['fp_length']))

			in_and_out_sizes = zip(all_layer_sizes[:-1], all_layer_sizes[1:])
			for layer, (N_prev, N_cur) in enumerate(in_and_out_sizes):
				setattr(self, 'layer_'+str(layer)+'_self_filter', L.Linear(N_prev, N_cur))
				for degree in degrees:
				   	name = weights_name(layer, degree)
					setattr(self, name, L.Linear(N_prev + num_bond_features(), N_cur))

	def __call__(self, x, y):
		return F.mean_squared_error(self.fwd(x),y)

	def fwd(self, smiles):
		array_rep = array_rep_from_smiles(tuple(smiles)) #rdkitで計算。smiles to data
	
		def update_layer(self, layer, atom_features, bond_features, array_rep, normalize=False):
			def get_weights_func(degree): #layer と degree からパラメータを選択する。
				return "self.layer_" + str(layer) + "_degree_" + str(degree) + "_filter"
			layer_self_weights = eval("self.layer_" + str(layer) + "_self_filter")

			self_activations = layer_self_weights(atom_features)
			neighbor_activations = (matmult_neighbors(self,
				array_rep, atom_features, bond_features, get_weights_func))

			total_activations = neighbor_activations + self_activations
			if normalize:
				total_activations = batch_normalize(total_activations)
			return F.relu(total_activations)

		def output_layer_fun_and_atom_activations(self, smiles):
			array_rep = array_rep_from_smiles(tuple(smiles))
			atom_features = array_rep['atom_features']
			bond_features = array_rep['bond_features']

			atom_features = bool_to_float32(atom_features)
			bond_features = bool_to_float32(bond_features)

			all_layer_fps = []
			atom_activations = []
			def write_to_fingerprint(self, atom_features, layer):
				cur_out_weights = eval("self.layer_output_weights_" + str(layer))
				atom_outputs = F.softmax(cur_out_weights(atom_features))
				atom_activations.append(atom_outputs)
				# Sum over all atoms within a moleclue:
				layer_output = sum_and_stack(atom_outputs, array_rep['atom_list'])
				all_layer_fps.append(layer_output)

			num_layers = self.model_params['fp_depth']
			for layer in xrange(num_layers):
				global atom_features
				write_to_fingerprint(self, atom_features, layer)
				atom_features = update_layer(self, layer, atom_features, bond_features, array_rep, normalize=False)
				atom_features = atom_features._data[0]

			write_to_fingerprint(self, atom_features, num_layers)
			return sum(all_layer_fps), atom_activations, array_rep
	
		def output_layer_fun(self, smiles):
			output, _, _ = output_layer_fun_and_atom_activations(self, smiles)
			return output
	
		def compute_atom_activations(self, smiles):
			_, atom_activations, array_rep = output_layer_fun_and_atom_activations(smiles)
			return atom_activations, array_rep
	
		conv_fp_func = output_layer_fun
		return (conv_fp_func(self, smiles))

