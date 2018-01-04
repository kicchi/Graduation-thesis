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

from NNFP import load_data
from NNFP import normalize_array
from NNFP import Deep_neural_network
from NNFP import Finger_print


task_params = {'target_name' : 'measured log solubility in mols per litre',
				'data_file'  : 'delaney.csv'}

N_train = 8
N_val   = 3
N_test  = 3

model_params = dict(fp_length = 50,      
					fp_depth = 4,       #NNの層と、FPの半径は同じ
					conv_width = 20,    #必要なパラメータはこれだけ（？）
					h1_size = 100,      #最上位の中間層のサイズ
					L2_reg = np.exp(-2))

train_params = dict(num_iters = 100,
					batch_size = 50,
					init_scale = np.exp(-4),
					step_size = np.exp(-6))

#FPを食わせるNNの定義
vanilla_net_params = dict(
	layer_sizes = [model_params['fp_depth'], model_params['h1_size']],
	normalize = True, 
	L2_reg = model_params['L2_reg']
	)

def weight_reg(self,weight, model_params):
	w = (F.reshape(weight, (weight.shape[0] * weight.shape[1], 1)))
	return  model_params['L2_reg'] * F.sum(w * w) / len(weight)
	
class Main(Chain):
	def __init__(self, model_params):
		super(Main, self).__init__(
			fp = Finger_print.FP(model_params),
			dnn = Deep_neural_network.DNN(model_params),
		)
	
	def __call__(self, x, y):
		y = Variable(np.array(y, dtype=np.float32))
		pred = self.prediction(x)
		return F.mean_squared_error(pred, y)

	def prediction(self, x):
		x = Variable(x)
		finger_print = self.fp(x)
		pred = self.dnn(finger_print)
		return pred
	
def mini_batche(x, y, batch_size, itr):
	start = (itr * batch_size) % len(x)
	end = (start + batch_size) % len(x)
	if start > end:
		end = len(x) - 1
	batch_x = x[start:end]
	batch_y = y[start:end]
	return batch_x, batch_y

def train_nn(model, train_smiles, train_raw_targets, seed=0,
				validation_smiles=None, validation_raw_targets=None):

	num_print_examples = N_train
	train_targets, undo_norm = normalize_array(train_raw_targets)
	training_curve = []
	optimizer = optimizers.Adam()
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))	
	
	for itr in range(100):
		x = train_smiles
		y = train_targets
		model.zerograds()
		batched_x, batched_y = mini_batche(x, y, train_params['batch_size'], itr)
		loss = model(batched_x, batched_y)
		#loss = model(x, y)
		if itr % 10 == 0:
			train_preds = undo_norm(model(train_smiles, train_targets))
			cur_loss = loss
			training_curve.append(cur_loss)
			print "Iteration", itr, "loss", cur_loss._data[0], \
				"train RMSE", (train_preds._data[0]),
			if validation_smiles is not None:
				validation_preds = undo_norm(model(validation_smiles, validation_raw_targets))
				print  "Validation RMSE", itr, ":", (validation_preds._data[0])
		loss.backward()
		optimizer.update()
		#print loss

		
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
	y_tests = np.reshape(y_tests, (N_test, 1)).astype(np.float32)

	def run_conv_experiment():
		conv_layer_sizes = [model_params['conv_width']] * model_params['fp_length']
		conv_arch_params = {'num_hidden_features' : conv_layer_sizes,  
							'fp_length' : model_params['fp_length'], 'normalize' : 1}
		'''Initialize model'''
		NNFP = Main(model_params) 
		optimizer = optimizers.Adam()
		optimizer.setup(NNFP)
		'''Learn'''
		trained_NNFP, conv_training_curve = \
			train_nn(NNFP, 
					x_trains, y_trains,  
					validation_smiles=x_vals, 
					validation_raw_targets=y_vals)
			
		'''Test'''
		#test_inputs is smiles
		test_inputs = x_tests
		test_targets = Variable(np.array(y_tests, dtype=np.float32))
		test_predictions = trained_NNFP.prediction(test_inputs)
	
		return F.mean_squared_error(test_predictions, test_targets)._data[0]

	print "Starting neural fingerprint experiment..."
	test_loss_neural = run_conv_experiment()
	print 
	print  "Neural test RMSE", test_loss_neural

if __name__ == '__main__':
	main()
