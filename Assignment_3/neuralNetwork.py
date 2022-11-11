from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

from activationFunc import activationFunctions as af
    
class neuralNetwork:
    def initialise(self,shape,initialiseType):
        if(initialiseType=="z"):
            return np.zeros(shape)
        elif(initialiseType=="r"):
            return np.random.rand(shape[0],shape[1])
        elif(initialiseType=="n"):
            return np.random.normal(0,1,shape)
    
    def __init__(self,n,layerSize,lr,activationFunc,initialWeights,epoch,batch):
        self.ao = af()
        self.n =n
        self.learningRate = lr
        self.activationFunc = activationFunc
        self.epoch = epoch
        self.batch = batch
        self.layerSize = layerSize
        self.initialWeights = initialWeights
        self.weightValues = []
        self.biasValues = []
        self.activationFuncValues = []
        self.deltaValues = []
        self.slopeValues = []
        self.trainCost = []
        self.validationCost = []
        self.accuracyValues = []
        self.initialiseWeightsandBias()

    def initialiseWeightsandBias(self):
        i=0
        while(i<self.n-1):
            if(self.initialWeights =="zero"):
                self.weightValues.append(self.initialise((self.layerSize[i],self.layerSize[i+1]),"z"))
            elif(self.initialWeights =="random"):
                self.weightValues.append(self.initialise((self.layerSize[i],self.layerSize[i+1]),"r"))
            elif(self.initialWeights =="normal"):
                self.weightValues.append(self.initialise((self.layerSize[i],self.layerSize[i+1]),"n"))
            i+=1
        i=0
        while(i<self.n-1):
            self.biasValues.append(np.zeros((1,self.layerSize[i+1])))
            i+=1
        
    def forwardPropagation(self,val):
        self.activationFuncValues = []
        self.activationFuncValues.append(val)
        i=0
        while(i<self.n-2):
            weightDot = np.dot(self.activationFuncValues[i],self.weightValues[i])
            if(self.activationFuncValues[i].shape[0]<self.biasValues[i].shape[0]):
                print(self.activationFuncValues[i].shape[0],self.biasValues[i].shape[0])
                break
            if(self.activationFunc == "linear"):
                self.activationFuncValues.append(self.ao.linear(weightDot+self.biasValues[i]))
            elif(self.activationFunc == "sigmoid"):
                self.activationFuncValues.append(self.ao.sigmoid(weightDot+self.biasValues[i]))
            elif(self.activationFunc == "tanh"):
                self.activationFuncValues.append(self.ao.tanh(weightDot+self.biasValues[i]))
            elif(self.activationFunc == "relu"):
                self.activationFuncValues.append(self.ao.relu(weightDot+self.biasValues[i]))
            elif(self.activationFunc == "leakyRelu"):
                self.activationFuncValues.append(self.ao.leakyRelu(weightDot+self.biasValues[i]))
            i += 1
        self.activationFuncValues.append(self.predictProb())

    def predictProb(self):
        return(self.ao.softmax(np.dot(self.activationFuncValues[self.n-2],self.weightValues[self.n-2])+self.biasValues[self.n-2]))

    def backwardPropogation(self,x,y):
        self.deltaValues = []
        self.deltaValues.append(self.activationFuncValues[self.n-1]-y)
        self.slopeValues = []
        i = self.n-2
        
        while(i>0):
            dot = np.dot(self.deltaValues[self.n-2-i],self.weightValues[i].T)
            if self.activationFunc == "linear":
                self.deltaValues.append(dot*self.ao.linear(self.activationFuncValues[i],True))
            elif self.activationFunc == "sigmoid":
                self.deltaValues.append(dot*self.ao.sigmoid(self.activationFuncValues[i],True))
            elif self.activationFunc == "tanh":
                self.deltaValues.append(dot*self.ao.tanh(self.activationFuncValues[i],True))
            elif self.activationFunc == "relu":
                self.deltaValues.append(dot*self.ao.relu(self.activationFuncValues[i],True))
            elif self.activationFunc == "leakyRelu":
                self.deltaValues.append(dot*self.ao.leakyRelu(self.activationFuncValues[i],True))
            i+=-1
        i = 0
        while(i<self.n-1):
            dot = np.dot(self.activationFuncValues[i].T,self.deltaValues[self.n-2-i])
            self.slopeValues.append(dot)
            i+=1
        
    def updatePar(self):
        i=0
        while(i<self.n-1):
            self.weightValues[i] =self.weightValues[i] - self.learningRate*self.slopeValues[i]
            i+=1
        i=0
        while(i<self.n-1):
            self.biasValues[i] = self.biasValues[i] -self.learningRate*self.deltaValues[self.n-2-i]
            i+=1
    
    def fit(self,x,y,valX,valY):
        i =0
        while(i<self.epoch):
            j=0
            while(j<x.shape[0]):
                if(j+self.batch < x.shape[0]):
                    self.forwardPropagation(x[j:j+self.batch])
                    self.backwardPropogation(x[j:j+self.batch],y[j:j+self.batch])
                    self.updatePar()
                j += self.batch
            accuracyTemp = []
            j=0
            while(j<x.shape[0]):
                if(j+self.batch<x.shape[0]):
                    self.forwardPropagation(x[j:j+self.batch])
                    accuracyTemp.append(self.accuracyScore(y[j:j+self.batch],self.activationFuncValues[self.n-1]))
                j += self.batch
            self.accuracyValues.append(np.mean(accuracyTemp))

            lossTemp = []
            j=0
            while(j<x.shape[0]):
                if(j+self.batch<x.shape[0]):
                    self.forwardPropagation(x[j:j+self.batch])
                    lossTemp.append(self.costFunc(y[j:j+self.batch],self.activationFuncValues[self.n-1]))
                j += self.batch
            self.trainCost.append(np.mean(lossTemp))

            lossTemp = []
            j=0
            while(j<valX.shape[0]):
                if(j+self.batch<valX.shape[0]):
                    self.forwardPropagation(valX[j:j+self.batch])
                    lossTemp.append(self.costFunc(valY[j:j+self.batch],self.activationFuncValues[self.n-1]))
                j += self.batch
            self.validationCost.append(np.mean(lossTemp))
            i+=1

    def accuracyScore(self,y,yPred):
        ans = 0
        i=0
        while(i<self.batch):
            if(np.argmax(yPred[i])==np.argmax(y[i])):
                ans+=1
            i+=1
        return(ans/self.batch)

    def costFunc(self,y,yPred):
        l = 0
        i=0
        while(i<y.shape[0]):
            l += -y[i]*np.log(yPred[i]+1e-15)
            i+=1
        temp = l/y.shape[0]
        return temp

    def predict(self,x):
        self.forwardPropagation(x)
        return(self.activationFuncValues[self.n-1])
    
    def costPlot(self):
        plt.plot(self.trainCost)
        plt.plot(self.validationCost)
        plt.xlabel("Epoch Value")
        plt.ylabel("Obtained Cost")
        plt.legend(["Validation Cost","Training Cost"])
        plt.show()
        








    


