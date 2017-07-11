import numpy
from NearestNeighbor import NearestNeighbor

Xtrain = numpy.zeros(shape=(50000, 32, 32, 3))
Ytrain = [numpy.random.randint(0, 10)] * 50000

Xtest = numpy.zeros(shape=(10000, 32, 32, 3))
Ytest = [numpy.random.randint(0, 10)] * 10000
nn = NearestNeighbor()

Xtr_rows = Xtrain.reshape(Xtrain.shape[0], 32 * 32 * 3)
Xte_rows = Xtest.reshape(Xtest.shape[0], 32 * 32 * 3)

# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50,000 x 3072 matrix
Xval_rows = Xtr_rows[:1000, :]  # take first 1000 for validation
Yval = Ytrain[:1000]
Xtr_rows = Xtr_rows[1000:, :]  # keep last 49,000 for train
Ytr = Ytrain[1000:]

# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:
    # use a particular value of k and evaluation on validation data
    nn = NearestNeighbor()
    nn.train(Xtr_rows, Ytr)
    # here we assume a modified NearestNeighbor class that can take a k as input
    Yval_predict = nn.predict(Xval_rows, k=k)
    acc = numpy.mean(Yval_predict == Yval)
    print(*'accuracy: %f' % (acc,))

    # keep track of what works on the validation set
    validation_accuracies.append((k, acc))