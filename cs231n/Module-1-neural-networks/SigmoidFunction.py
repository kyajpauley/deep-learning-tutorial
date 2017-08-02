import math

wVals = [2, -3, -3] # assume some random weights and data
xVals = [-1, -2]

# forward pass
x = wVals[0] * xVals[0] + wVals[1] * xVals[1] + wVals[2]  # this is a really weird way to get the dot product of something, but it
                                    # at least shows the math?
f = 1.0 / (1 + math.exp(-x)) # sigmoid function
# f = 1/(1 + e**−x)

# backward pass through the neuron (backpropagation)
ddot = (1 - f) * f # gradient on dot variable, using the sigmoid gradient derivation
dx = [wVals[0] * ddot, wVals[1] * ddot] # backprop into x
dw = [xVals[0] * ddot, xVals[1] * ddot, 1.0 * ddot] # backprop into w
# we're done! we have the gradients on the inputs to the circuit