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

class Mymodel(Chain):
	def __init__(self, model_params):
		super(Mymodel, self).__init__()
		with self.init_scope():
			setattr(self, 'h1', L.Linear(2,3))	
			setattr(self, 'h2', L.Linear(3,1))	

	def __call__(self, x, y):
		return F.mean_squared_name

model_params = dict(fp_length = 50,      
					fp_depth = 4,       #NNの層と、FPの半径は同じ
					conv_width = 20,    #必要なパラメータはこれだけ（？）
					h1_size = 100,      #最上位の中間層のサイズ
					L2_reg = np.exp(-2))



model = Mymodel(model_params)


x = np.array(range(2), dtype = np.float32)
print x
print type(x)
y = np.array(range(3), dtype = np.float32)
xv = Variable(x)
y = [1,2,3]


str1 = 'model.h1(xv)'
print eval(str1)
#print type(model.(eval(str1)))
