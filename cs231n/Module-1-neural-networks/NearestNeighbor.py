import numpy as np


class NearestNeighbor(object):
    def __init__(self):
        self.Xtr = []
        self.ytr = []

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        # loop over all test rows
        for i in range(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            # distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1) # L1 distance = d1(I1,I2)=∑p|Ip1−Ip2|
            distances = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis=1)) # L2 distance = d2(I1,I2)= sqrt(∑p((Ip1−Ip2)**2)) euclidian distance
            min_index = np.argmin(distances)  # get the index with smallest distance
            Ypred[i] = self.ytr[min_index]  # predict the label of the nearest example

        return Ypred
