import numpy as np
class activationFunctions:
    def sigmoid(self,x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))
    def tanh(self,x,deriv=False):
        if(deriv==True):
            return 1-np.tanh(x)**2
        return np.tanh(x)
    def relu(self,x,deriv=False):
        if(deriv==True):
            x[x<=0] = 0
            x[x>0] = 1
            return x
        return np.maximum(x,0)
    def leakyRelu(self,x,deriv=False):
        if(deriv==True):
            x[x<=0] = 0.01
            x[x>0] = 1
            return x
        return np.maximum(0.01*x,x)
    def softmax(self,x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        max = np.max(x)
        exponents = np.exp(x-max)
        return exponents/np.sum(exponents)
    
    def linear(self,x,deriv=False):
        if(deriv==True):
            return(np.ones(x.shape))
        return x