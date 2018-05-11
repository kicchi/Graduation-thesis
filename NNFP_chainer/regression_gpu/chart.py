import math
import numpy as np
from matplotlib import pyplot

'''
pi = math.pi   

x = np.linspace(0, 2*pi, 100)  
y = np.sin(x)

'''

x = np.linspace(0, 2000, 10)  
curve = np.array([input() for _ in range(10)])


pyplot.plot(x, curve)
pyplot.show()
