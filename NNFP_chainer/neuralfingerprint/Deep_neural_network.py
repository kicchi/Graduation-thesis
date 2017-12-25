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


'''
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data.astype(np.float32)
Y = iris.target
N = Y.size
Y2 = np.zeros(3*N).reshape(N,3).astype(np.float32)
for i in range(N):
	Y2[i,Y[i]] = 1.0

index = np.arange(N)
xtrain = X[index[index%2!=0],:]
ytrain = Y2[index[index%2!=0],:]
xtest = X[index[index%2==0],:]
yans = Y[index[index%2==0]]
'''

class DNN(Chain):
	def __init__(self, model_params):
		unit_num = 100
		super(DNN, self).__init__(
			l1 = L.Linear(model_params['fp_length'],unit_num),
			l2 = L.Linear(unit_num,1),
		)

	def __call__(self, x, y):
		return F.mean_squared_error(self.fwd(x),y)

	def fwd(self, x):
		h = F.relu(self.l1(x))
		o = self.l2(h)
		return o

'''
fp_depth = 4
fp_length = 3
fp = DNN()
optimizer = optimizers.Adam()
optimizer.setup(fp)
	
for i in range(10000):
	x = Variable(xtrain)
	y = Variable(ytrain)
	fp.zerograds()
	loss = fp(x,y)
	if i % 100 == 0:
		print "loss, ", loss
	loss.backward()
	optimizer.update()

xt = Variable(xtest)
yy = fp.fwd(xt)


ans = yy.data
nrow, ncol = ans.shape
ok = 0

for i in range(nrow):
	cls = np.argmax(ans[i,:])
	print ans[i, :], cls
	if cls == yans[i]:
		ok += 1
	
print ok, "/", nrow, " = ", (ok * 1.0) / nrow 


				


'''
