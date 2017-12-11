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
from sklearn.datasets import fetch_mldata

#from chainer_nn import load_data
#from chainer_nn import build_conv_deep_net
#from chainer_nn import normalize_array, adam
#from chainer_nn import build_batched_grad
#from chainer_nn.utils import  rmse


#set data
train, test = chainer.datasets.get_mnist()
x_train, y_train = train._datasets
x_test, y_test = test._datasets

N_train = 1000
N_test = 30
x_train = x_train[0: N_train]
y_train = y_train[0: N_train]
x_test = x_test[0: N_test]
y_test = y_test[0: N_test]

#Define model
input_size = len(x_train[0])
batchsize = 100
n_epoch = 20
n_units = 1000



class DNN(Chain,input_size):
	def __init__(self, input_size):
		super(DNN, self).__init__(
			l1 = L.Linear(input_size, 1000),
			l2 = L.Linear(1000, 10),
		)

	def __call__(self, x, y):
		return F.softmax_cross_entropy(self.fwd(x), y) #rmse

	def fwd(self, x):
		h1 = F.relu(self.l1(x)) 
		return h1


#initialize mdoel

model = DNN(input_size)
optimizer = optimizers.Adam()
optimizer.setup(model)

#learn

print "learning..."
for i in range(1000):
	x = Variable(x_train)
	y = Variable(y_train)
	model.zerograds()
	loss = model(x, y)
	loss.backward()
	optimizer.update()
	if i % 100 == 0:
		print i, " train", " loss = " , loss

#test
xt = Variable(x_test)
yy = model.fwd(xt)
ans = yy.data

print y_test
ok = 0
for i in range(N_test):
	cls = np.argmax(ans[i], axis=0)
	if y_test[i] == cls:
		ok += 1

print ok, "/", N_test, " = " , (ok * 1.0) / N_test
