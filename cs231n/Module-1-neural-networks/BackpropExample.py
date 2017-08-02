import math

# fake function
# f(x,y) = (x + σ(y))/(σ(x) + (x + y)**2)

x = 3 # example values
y = -4

# forward pass
sigmoid_y = 1.0 / (1 + math.exp(-y)) # sigmoid in numerator                         #(1)
# f = 1/(1 + e**−x)
numerator = x + sigmoid_y # numerator (x + σ(y))                                    #(2)
sigmoid_x = 1.0 / (1 + math.exp(-x)) # sigmoid in denominator                       #(3)
sum_xy = x + y                                                                      #(4)
sum_xy_squared = sum_xy ** 2                                                        #(5)
denominator = sigmoid_x + sum_xy_squared # denominator                              #(6)
inverse_denominator = 1.0 / denominator                                             #(7)
f = numerator * inverse_denominator # done!                                         #(8)

# work backwards through broken down steps to find derivatives
# backprop f = numerator * inverse_denominator
d_numerator = inverse_denominator # gradient on numerator                           #(8)
d_inverse_denominator = numerator                                                   #(8)
# backprop inverse_denominator = 1.0 / denominator
d_denominator = (-1.0 / (denominator ** 2)) * d_inverse_denominator                 #(7)
# backprop denenominator = sigmoid_x + sum_xy_squared
d_sigmoid_x = (1) * d_denominator                                                   #(6)
d_sum_xy_squared = (1) * d_denominator                                              #(6)
# backprop sum_xy_squared = sum_xy**2
d_sum_xy = (2 * sum_xy) * d_sum_xy_squared                                          #(5)
# backprop sum_xy = x + y
d_x = (1) * d_sum_xy                                                                #(4)
d_y = (1) * d_sum_xy                                                                #(4)
# backprop sigmoid_x = 1.0 / (1 + math.exp(-x))
d_x += ((1 - sigmoid_x) * sigmoid_x) * d_sigmoid_x # Notice += !! See notes below   #(3)
# backprop numerator = x + sigmoid_y
d_x += (1) * d_numerator                                                            #(2)
d_sigmoid_y = (1) * d_numerator                                                     #(2)
# backprop sigmoid_y = 1.0 / (1 + math.exp(-y))
d_y += ((1 - sigmoid_y) * sigmoid_y) * d_sigmoid_y                                  #(1)
# done! phew