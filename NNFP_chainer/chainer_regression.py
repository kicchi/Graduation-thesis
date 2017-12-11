#coding: utf-8
import numpy as np
import numpy.random as npr
#import cupy as cp #GPUを使うためのnumpy
import chainer 
from chainer import cuda, Function, gradient_check, \
	Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.function as F
import chainer.links as L

from chainer_nn import load_data
from chainer_nn import build_conv_deep_net
from chainer_nn import normalize_array, adam
from chainer_nn import build_batched_grad
from chainer_nn.utils import  rmse


task_params = {'target_name' : 'measured log solubility in mols per litre',
				'data_file'  : 'delaney.csv'}

N_train = 8
N_val   = 2
N_test  = 2

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

def train_nn(pred_fun, loss_fun, num_weights, train_smiles, train_raw_targets, train_params, seed = 0, validation_smiles = None, validation_raw_targets = None):
	print "Total number of weights in the network:", num_weights
	init_weights = npr.RandomState(seed).randn(num_weights) * train_params['init_scale']
	num_print_examples = 100
	train_targets, undo_norm = normalize_array(train_raw_targets)
	training_curve = []

	#autograd の grad() をchainer で書き直す

def main():
	print "Loading data..."
	traindata, valdata, testdata = load_data(
		task_params['data_file'],
		(N_train, N_val, N_test),
		input_name = 'smiles', 
		target_name = task_params['target_name'])

	train_inputs, train_targets = traindata
	val_inputs,     val_targets = valdata
	test_inputs,   test_targets = testdata

	def print_performance(pred_func):
		train_pred = pred_func(train_inputs)
		val_preds = pred_func(val_inputs)
		print "\nPerformance (RMSE) on " + task_params['target_name'] + ":"
		print "Train:", rmse(train_preds, train_targets) 
		print "Test: ", rmse(val_preds, val_targets)
		print "-" * 80
		return rmse(val_preds, val_targets)

	def run_conv_experiment():
		conv_layer_sizes = [model_params['conv_width']] * model_params['fp_depth']
		conv_arch_params = {'num_hidden_features' : conv_layer_sizes,
							'fp_length' : model_params['fp_length'], 'normalize' : 1}
		#モデルの定義
		loss_fun, pred_fun, conv_parser = \
			build_conv_deep_net(conv_arch_params, vanilla_net_params, model_params['L2_reg'])
		num_weights = len(conv_parser)

		#モデルの学習
		predict_func, trained_weights, conv_training_curve = \
			train_nn(pred_fun, loss_fun, num_weights, train_inputs, train_targets,
					 train_params, validation_smiles = val_inputs, validation_raw_targets = val_targets)

		test_predictions = predict_func(test_inputs)
		return rmse(test_predictions, test_targets)

	print "Task params", task_params
	print 
	print "Starting neural fingerprint experiment..."
	test_loss_neural = run_conv_experiment()
	print 
	print "Neural test RMSE:", test_loss_neural

if __name__ == '__main__':
	main()
