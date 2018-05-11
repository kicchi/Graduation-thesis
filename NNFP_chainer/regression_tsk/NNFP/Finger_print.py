#coding: utf-8
import numpy as np
#import cupy as cp #GPUを使うためのnumpy
import chainer 
from chainer import cuda, Function, Variable, optimizers, initializers
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L
from collections import OrderedDict
from features import num_atom_features, num_bond_features
from mol_graph import graph_from_smiles_tuple, degrees
from time import time

def fast_array_from_list(xs):
    fast_array = Variable(np.empty((0,len(xs[0])), dtype=np.float32))
    for x in xs:
        fast_array = F.concat((fast_array, (F.expand_dims(x, axis=0))), axis=0)
    return fast_array

def sum_and_stack(features, idxs_list_of_lists):
    return fast_array_from_list([F.sum(features[idx_list], axis=0) for idx_list in idxs_list_of_lists])

def graph_from_smiles_tuple(smiles_tuple): #使わない
    graph_list = [graph_from_smiles(s) for s in smiles_tuple]
    big_graph = MolGraph()
    for subgraph in graph_list:
        big_graph.add_subgraph(subgraph)

    # This sorting allows an efficient (but brittle!) indexing later on.
    big_graph.sort_nodes_by_degree('atom')
    return big_graph

def graph_from_smiles_one(smiles):
	graph = graph_from_smiles(s)
	big_graph = MolGraph()
	big_graph.add_subgraph(graph)
	return big_graph

def array_rep_from_smiles(smiles):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    molgraph = graph_from_smiles_one(smiles)
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'), # List of lists.
                'rdkit_ix'      : molgraph.rdkit_ix_array()}  # For plotting only.
				

    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep


def matmult_neighbors(self, array_rep, atom_features, bond_features, get_weights_func):
	activations_by_degree = np.empty((0,20), dtype=np.float32)
	for degree in degrees:
		get_weights = eval(get_weights_func(degree))
		atom_neighbors_list = array_rep[('atom_neighbors', degree)]
		bond_neighbors_list = array_rep[('bond_neighbors', degree)]
		if len(atom_neighbors_list) > 0:
			neighbor_features = [atom_features[atom_neighbors_list],
								bond_features[bond_neighbors_list]]
			stacked_neighbors = np.concatenate(neighbor_features, axis=2)
			summed_neighbors = F.sum(stacked_neighbors,axis=1)
  			activations = get_weights(summed_neighbors)
			activations_by_degree = F.concat((activations_by_degree, activations), axis=0)
	return activations_by_degree

def weights_name(layer, degree):
    return "layer_" + str(layer) + "_degree_" + str(degree) + "_filter"

def bool_to_float32(features):
	return np.array(features).astype(np.float32)

def bool_to_float32_one_dim(features):
	vec = np.empty((0,1), dtype=np.float32)
	for f in features:
		new_idx = 0.0
		for idx in range(len(f)):
			if f[idx]:
				new_idx += idx
		vec = np.append(vec, np.array([[new_idx]], dtype=np.float32),axis=0)
	return vec


def build_weights(self, model_params):
	initializer = chainer.initializers.HeNormal() 
	setattr(self, 'model_params', model_params,)
	num_hidden_features = [self.model_params['conv_width']] * self.model_params['fp_depth']
	all_layer_sizes = [num_atom_features()] + num_hidden_features

	'''output weights'''
	for layer in range(len(all_layer_sizes)):
		setattr(self, 'layer_output_weights_'+str(layer), L.Linear(all_layer_sizes[layer], self.model_params['fp_length'], initialW=initializer))

	'''hidden weights'''
	in_and_out_sizes = zip(all_layer_sizes[:-1], all_layer_sizes[1:])
	for layer, (N_prev, N_cur) in enumerate(in_and_out_sizes):
		setattr(self, 'layer_'+str(layer)+'_self_filter', L.Linear(N_prev, N_cur,initialW=initializer))
		for degree in degrees:
		   	name = weights_name(layer, degree)
			setattr(self, name, L.Linear(N_prev + num_bond_features(), N_cur,initialW=initializer))

class FP(Chain):
	def __init__(self, model_params):
		super(FP, self).__init__()
		with self.init_scope():
			build_weights(self, model_params)

	def __call__(self, array_rep):
		#array_rep = array_rep_from_smiles(tuple(smiles)) #rdkitで計算。smiles to data
	
		def update_layer(self, layer, atom_features, bond_features, array_rep, normalize=False):
			def get_weights_func(degree): #layer と degree からパラメータを選択する。
				return "self.layer_" + str(layer) + "_degree_" + str(degree) + "_filter"
			layer_self_weights = eval("self.layer_" + str(layer) + "_self_filter")

			self_activations = layer_self_weights(atom_features)
			neighbor_activations = matmult_neighbors(self,
				array_rep, atom_features, bond_features, get_weights_func)
			total_activations = neighbor_activations + self_activations
			if normalize: #FPでbatch normalizationいらない
				total_activations = F.batch_normalization(total_activations)
			return F.relu(total_activations)

		def output_layer_fun_and_atom_activations(self, array_rep):
			#array_rep = array_rep_from_smiles(tuple(smiles))
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
				write_to_fingerprint(self, atom_features, layer)
				atom_features = update_layer(self, layer, atom_features, bond_features, array_rep, normalize=False)
				atom_features = atom_features._data[0]

			write_to_fingerprint(self, atom_features, num_layers)
			return sum(all_layer_fps), atom_activations, array_rep
	
		def output_layer_fun(self, array_rep):
			output, _, _ = output_layer_fun_and_atom_activations(self, array_rep)
			return output
	
		conv_fp_func = output_layer_fun
		return (conv_fp_func(self, array_rep))

