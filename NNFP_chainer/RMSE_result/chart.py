#coding : utf-8
import math
import numpy as np
from matplotlib import pyplot

'''
pi = math.pi   

x = np.linspace(0, 2*pi, 100)  
y = np.sin(x)

'''

x = np.linspace(0, 2000, 20)  
ecfp = np.array([input() for _ in range(20)])
fcfp = np.array([input() for _ in range(20)])
#one_dim = np.array([input() for _ in range(10)])
ecfc = np.array([input() for _ in range(20)])
attention = np.array([input() for _ in range(20)])


pyplot.plot(x, ecfp, label = 'feature representation 1')
pyplot.plot(x, fcfp, label = 'feature representation 2')
#pyplot.plot(x, one_dim, label = "atomic number")
pyplot.plot(x, ecfc, label = "feature representation 1,2(not attention)")
pyplot.plot(x, attention, label = "feature representaiton 1,2(attention)")

pyplot.title("data A")

pyplot.xlabel("learn times")
pyplot.ylabel("RMSE")



pyplot.legend()

pyplot.show()
