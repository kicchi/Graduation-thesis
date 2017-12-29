#coding: utf-8
#import cupy as cp #GPUを使うためのnumpy
import numpy as np
import chainer 
from chainer import Link, Chain, Variable
import chainer.functions as F
#import chainer.Summary as S
import chainer.links as L
from  util import normalize_array

class DNN(Chain):
	def __init__(self, model_params):
		super(DNN, self).__init__(
			l1 = L.Linear(model_params['fp_length'],model_params['h1_size']),
			l2 = L.Linear(model_params['h1_size'],1),
		)

	def __call__(self, x):
		h = self.l1(x)
		o = self.l2(h)
		#print "o ", o
		#print "n_o ", self.normalize(o)
		#return self.normalize(o)
		return o

	def normalize(self, x):
		
		#print x
		x = x._data[0]
		sum_x = 0
		for i in x:
			sum_x += i
		mean = sum_x / len(x)
		
		#stdの計算
		sum_x = 0
		for i in x:
			sum_x += (i - mean) ** 2	
		std = (sum_x / len(x)) ** (0.5)
		x = (x - mean) / (std + 1)
	
		return Variable(x)

		
