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

from neuralfingerprint import load_data
from neuralfingerprint import build_conv_deep_net
from neuralfingerprint import normalize_array, adam
from neuralfingerprint import build_batched_grad
from neuralfingerprint import Deep_neural_network
from neuralfingerprint import Finger_print
from neuralfingerprint.utils import  rmse


task_params = {'target_name' : 'measured log solubility in mols per litre',
				'data_file'  : 'delaney.csv'}

N_train = 800
N_val   = 20
N_test  = 20

model_params = dict(fp_length = 50,      
					fp_depth = 4,       #NNの層と、FPの半径は同じ
					conv_width = 20,    #必要なパラメータはこれだけ（？）
					h1_size = 100,      #最上位の中間層のサイズ
					L2_reg = np.exp(-2))

train_params = dict(num_iters = 100,
					batch_size = 100,
					init_scale = np.exp(-4),
					step_size = np.exp(-6))

#FPを食わせるNNの定義
vanilla_net_params = dict(
	layer_sizes = [model_params['fp_depth'], model_params['h1_size']],
	normalize = True, 
	L2_reg = model_params['L2_reg']
	#nll_func = rmse  ToDo
	)

	
class Main(Chain):
	def __init__(self, model_params):
		super(Main, self).__init__(
			dnn = Deep_neural_network.DNN(model_params),
			fp = Finger_print.FP(model_params),
		)
	
	def __call__(self, x, y):
		finger_print = self.fp.fwd(x)
		output = self.dnn.fwd(finger_print)
		check = np.array(1)
		if type(check) is type(y):
			y = Variable(y)
		return F.mean_squared_error(output, y._data[0].astype(np.float32))
	

def train_nn(model, num_weights, train_smiles, train_raw_targets, seed=0,
				validation_smiles=None, validation_raw_targets=None):
	#init_weight
	num_print_examples = N_train
	train_targets, undo_norm = normalize_array(train_raw_targets)
	training_curve = []
	optimizer = optimizers.Adam()
	optimizer.setup(model)
	for itr in range(100):
		x = Variable(train_smiles)
		y = Variable(train_targets) #raw_targetsは使わない？正規化は平均０、分散１にする操作
		model.zerograds()
		loss = model(x, y)
		if itr % 10 == 0:
			train_preds = undo_norm(model(train_smiles[:num_print_examples], train_targets[:num_print_examples]))
			cur_loss = loss
			training_curve.append(cur_loss)
			print "Iteration", itr, "loss", cur_loss._data[0], \
				"train RMSE", (train_preds._data[0]),
			if validation_smiles is not None:
				validation_preds = undo_norm(model(validation_smiles[:N_val], validation_raw_targets[:N_val]))
				print  "Validation RMSE", itr, ":", (validation_preds._data[0])
		loss.backward()
		optimizer.update()
		
	return model, training_curve

def main():
	print "Loading data..."
	traindata, valdata, testdata = load_data(
		task_params['data_file'], (N_train, N_val, N_test),
		input_name = 'smiles', target_name = task_params['target_name'])
	x_trains, y_trains = traindata
	x_vals, y_vals = valdata
	x_tests, y_tests = testdata
	x_trains = np.reshape(x_trains, (N_train, 1))
	y_trains = np.reshape(y_trains, (N_train, 1))
	x_vals = np.reshape(x_vals, (N_val, 1))
	y_vals = np.reshape(y_vals, (N_val, 1))
	x_tests = np.reshape(x_tests, (N_test, 1))
	y_tests = np.reshape(y_tests, (N_test, 1))

	def run_conv_experiment():
		conv_layer_sizes = [model_params['conv_width']] * model_params['fp_length']
		conv_arch_params = {'num_hidden_features' : conv_layer_sizes,  
							'fp_length' : model_params['fp_length'], 'normalize' : 1}
		'''Initialize model'''
		NNFP = Main(model_params) #parameter is given to Main(...)
		optimizer = optimizers.Adam()
		optimizer.setup(NNFP)

		#num_weights = len(conv_parser) パラメータの総数後で出す
		num_weights = 32591
		'''Learn'''
		trained_NNFP, conv_training_curve = \
			train_nn(NNFP, num_weights, 
					x_trains, y_trains,  
					validation_smiles=x_vals, 
					validation_raw_targets=y_vals)
			
		'''Test'''
		#test_inputs is smiles
		test_inputs = Variable(x_tests)
		test_targets = Variable(y_tests)
		test_predictions = trained_NNFP(test_inputs, test_targets)
	
		#type is correspond?
		return test_predictions._data[0]

	print "Starting neural fingerprint experiment..."
	test_loss_neural = run_conv_experiment()
	print 
	print  "Neural test RMSE", test_loss_neural

if __name__ == '__main__':
	main()
