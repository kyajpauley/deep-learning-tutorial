# f(x,y,z)=(x+y)z ---> q=x+y and f=qz
# set some inputs
x = -2
y = 5
z = -4

# perform the forward pass
q = x + y  # q becomes 3
f = q * z  # f becomes -12

# perform the backward pass (backpropagation) in reverse order:
# first backprop through f = q * z
dfdz = q  # df/dz = q, so gradient on z becomes 3
dfdq = z  # df/dq = z, so gradient on q becomes -4
# now backprop through q = x + y
dfdx = 1.0 * dfdq  # dq/dx = 1. And the multiplication here is the chain rule!
dfdy = 1.0 * dfdq  # dq/dy = 1
