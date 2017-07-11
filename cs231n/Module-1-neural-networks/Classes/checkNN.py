import numpy
from NearestNeighbor import NearestNeighbor

Xtrain = numpy.zeros(shape=(32, 32, 3)) * 50000
Ytrain = [numpy.random.randint(0, 10)] * 50000

Xtest = numpy.zeros(shape=(32, 32, 3)) * 10000
Ytest = [numpy.random.randint(0, 10)] * 10000
nn = NearestNeighbor()

