import numpy
from NearestNeighbor import NearestNeighbor

Xtrain = numpy.zeros(shape=(50000, 32, 32, 3))
Ytrain = [numpy.random.randint(0, 10)] * 50000

Xtest = numpy.zeros(shape=(50000, 32, 32, 3))
Ytest = [numpy.random.randint(0, 10)] * 10000
nn = NearestNeighbor()

Xtr_rows = Xtrain.reshape(Xtrain.shape[0], 32 * 32 * 3)
Xte_rows = Xtest.reshape(Xtest.shape[0], 32 * 32 * 3)