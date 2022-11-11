import numpy as np


class activationFunctions:
    def sigmoid(self, x, deriv=False):
        if(deriv != True):
            expo = np.exp(-x)
            return 1/(1+expo)
        return -1*x*(x-1)

    def tanh(self, x, deriv=False):
        if(deriv != True):
            tanx = np.tanh(x)
            return tanx
        return 1-np.tanh(x)**2

    def relu(self, x, deriv=False):
        if(deriv != True):
            max = np.maximum(x, 0)
            return max

        x[x > 0] = 1
        x[x <= 0] = 0
        return x

    def leakyRelu(self, x, deriv=False):
        if(deriv != True):
            max = np.maximum(x, 0.01*x)
            return max
        x[x <= 0] = 0.01
        x[x > 0] = 1
        return x

    def softmax(self, x, deriv=False):
        if(deriv != True):
            max = np.max(x)
            exponents = np.exp(x-max)
            return exponents/np.sum(exponents)
        return x*(1-x)

    def linear(self, x, deriv=False):
        if(deriv != True):
            return x
        return(np.ones(x.shape))
