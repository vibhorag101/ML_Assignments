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
        self.trainCost = []
        self.weightValues = []
        self.biasValues = []
        self.epoch = epoch
        self.validationCost = []
        self.accuracyValues = []
        self.learningRate = lr
        self.batch = batch
        self.layerSize = layerSize
        self.deltaValues = []
        self.initialWeights = initialWeights
        self.activationFunc = activationFunc
        self.activationFuncValues = []
        self.n =n
        self.slopeValues = []
        self.ao = af()
        self.initialiseWeightsandBias()

    def initialiseWeightsandBias(self):
        i=0
        
        while(i<self.n-1):
            sizeTup = (self.layerSize[i],self.layerSize[i+1])
            if(self.initialWeights =="random"):
                self.weightValues.append(self.initialise(sizeTup,"r"))
            if(self.initialWeights =="zero"):
                self.weightValues.append(self.initialise(sizeTup,"z"))
            if(self.initialWeights =="normal"):
                self.weightValues.append(self.initialise(sizeTup,"n"))
            i+=1
        i=0
        while(i<self.n-1):
            zeroList = np.zeros((1,self.layerSize[i+1]))
            self.biasValues.append(zeroList)
            i+=1
        
    def forwardPropagation(self,val):
        self.activationFuncValues = []
        self.activationFuncValues.append(val)
        i=0
        while(i<self.n-2):
            if(self.biasValues[i].shape[0]>self.activationFuncValues[i].shape[0]):
                break
            weightDot = np.dot(self.activationFuncValues[i],self.weightValues[i])
            if(self.activationFunc == "linear"):
                self.activationFuncValues.append(self.ao.linear(weightDot+self.biasValues[i]))
            elif(self.activationFunc == "sigmoid"):
                self.activationFuncValues.append(self.ao.sigmoid(weightDot+self.biasValues[i]))
            elif(self.activationFunc == "tanh"):
                self.activationFuncValues.append(self.ao.tanh(weightDot+self.biasValues[i]))
            elif(self.activationFunc == "relu"):
                self.activationFuncValues.append(self.ao.relu(np.dot(self.activationFuncValues[i],self.weightValues[i])+self.biasValues[i]))
            elif(self.activationFunc == "leakyRelu"):
                self.activationFuncValues.append(self.ao.leakyRelu(weightDot+self.biasValues[i]))
            i += 1
        self.activationFuncValues.append(self.predictProb())

    def predictProb(self):
        dot = np.dot(self.activationFuncValues[self.n-2],self.weightValues[self.n-2])
        return(self.ao.softmax(dot+self.biasValues[self.n-2]))

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
            sub = self.learningRate*self.deltaValues[self.n-2-i]
            self.biasValues[i] = self.biasValues[i] - sub
            i+=1

        i=0
        while(i<self.n-1):
            sub = self.learningRate*self.slopeValues[i]
            self.weightValues[i] =self.weightValues[i] - sub
            i+=1

    
    def fit(self,x,y,valX,valY):
        i =0
        while(i<self.epoch):
            j=0
            while(j<x.shape[0]):
                if(x.shape[0] > j+self.batch):
                    xGive = x[j:j+self.batch]
                    yGive = y[j:j+self.batch]
                    self.forwardPropagation(xGive)
                    self.backwardPropogation(xGive,yGive)
                    self.updatePar()
                j += self.batch
            accuracyTemp = []
            j=0
            while(j<x.shape[0]):
                if(j+self.batch<x.shape[0]):
                    xGive = x[j:j+self.batch]
                    yGive = y[j:j+self.batch]
                    self.forwardPropagation(xGive)
                    accuracyTemp.append(self.accuracyScore(yGive,self.activationFuncValues[self.n-1]))
                j += self.batch
            self.accuracyValues.append(np.mean(accuracyTemp))

            lossTemp = []
            j=0
            while(j<x.shape[0]):
                if(j+self.batch<x.shape[0]):
                    xGive = x[j:j+self.batch]
                    yGive = y[j:j+self.batch]
                    self.forwardPropagation(xGive)
                    lossTemp.append(self.costFunc(yGive,self.activationFuncValues[self.n-1]))
                j += self.batch
            self.trainCost.append(np.mean(lossTemp))

            lossTemp = []
            j=0
            while(j<valX.shape[0]):
                if(j+self.batch<valX.shape[0]):
                    xGive = valX[j:j+self.batch]
                    yGive = valY[j:j+self.batch]
                    self.forwardPropagation(xGive)
                    lossTemp.append(self.costFunc(yGive,self.activationFuncValues[self.n-1]))
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
            logVal = np.log(yPred[i]+1e-15)
            l = l-y[i]* logVal
            i+=1
        temp = l/y.shape[0]
        return temp

    def predict(self,x):
        self.forwardPropagation(x)
        ans = self.activationFuncValues[self.n-1]
        return(ans)
    
    def costPlot(self):
        plt.plot(self.validationCost)
        plt.plot(self.trainCost)
        plt.ylabel("Obtained Cost")
        plt.xlabel("Epoch Value")
        plt.legend(["Validation Cost","Training Cost"])
        plt.show()
        








    


