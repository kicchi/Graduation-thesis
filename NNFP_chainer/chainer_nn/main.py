from  utils import collect_test_losses
from  utils import WeightsParser
import numpy as np

parser = WeightsParser()
parser.add_weights('atom', [2, 2])

print parser.N
print parser.idxs_and_shapes 

vect = np.array(range(2))
parser.get(vect, 'atom')
print vect
