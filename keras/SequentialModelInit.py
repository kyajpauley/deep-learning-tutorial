from keras.models import Sequential
from keras.layers import Dense, Activation

# declare model by passing list of layer instances
model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),  # doesn't have input shape since isn't first layer
    Activation('softmax'),
])

# or declare model and add layers after init
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))

# input shape specification (following snippets are equivalent for 2d layers)
# only for first layers! subsequent layers don't get this!
model = Sequential()
model.add(Dense(32, input_shape=(784,)))  # pass tuple of shape

model = Sequential()
model.add(Dense(32, input_dim=784))
