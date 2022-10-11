import numpy as np
import random

class Circle:
    def __init__(self,h,k,r,dataLabel):
        self.dataLabel = dataLabel
        self.h = h
        self.k = k
        self.r = r

    def getY(self,x,isPositive):
        if isPositive:
            return (abs(self.r**2 - (self.h-x)**2))**0.5 + self.k
        else:
            return -(abs(self.r**2 - (self.h-x)**2))**0.5 + self.k    
    
    def getDataPoints(self,mean=0,sd=0.1,noise=False):
        x = random.uniform(self.h-self.r,self.h+self.r)
        c = random.randint(0,1)
        y = -1
        if c == 0:
            y = self.getY(x,True)
        else:
            y = self.getY(x,False)
        
        if noise==False:
            return [x,y]
        else:
            return [random.gauss(mean,sd)+x,random.gauss(mean,sd)+y]


class DataSet:
    def __init__(self,numData):
        self.numData = numData
        self.c1 = Circle(0,0,1,0)
        self.c2 = Circle(0,3,1,1)

    def getDataNoiseFalse(self):
        points = []
        dataLabels = []
        i=0
        while i<self.numData:
            c = random.randint(0,1)
            if c == 0:
                tempLabel = self.c1.dataLabel
                tempPoint = self.c1.getDataPoints()
                dataLabels.append(tempLabel)
                points.append(tempPoint)
            else:
                tempLabel = self.c2.dataLabel
                tempPoint = self.c2.getDataPoints()
                dataLabels.append(tempLabel)
                points.append(tempPoint)
                
            i+=1
        ans = [points,dataLabels]
        return ans

    def getDataNoiseTrue(self):
            points = []
            dataLabels = []
            i=0
            while i<self.numData:
                c = random.randint(0,1)
                if c == 0:
                    tempLabel = self.c1.dataLabel
                    tempPoint = self.c1.getDataPoints(noise=True)
                    dataLabels.append(tempLabel)
                    points.append(tempPoint)   
                else:
                    tempLabel = self.c2.dataLabel
                    tempPoint = self.c2.getDataPoints(noise=True)
                    dataLabels.append(tempLabel)
                    points.append(tempPoint)

                i += 1
            ans = [points,dataLabels]
            return ans


    def getData(self,noise=False):
        points = []
        dataLabels = []
        if(noise==False):
            return self.getDataNoiseFalse()

        else:
            return self.getDataNoiseTrue()
                    

class MyPerceptron:
    def __init__(self,points,labels,biasState=True):
        self.points = points
        self.labels = labels
        self.biasState = biasState
        self.modelWeights = np.zeros(2)
        self.bias = 0
        self.iter = 1000
    def setDataPoints(self):
        delta = 0
        i=0
        combine = zip(self.points,self.labels)
        while i<len(combine):
            dataPoint = combine[i][0]
            dataLabel = combine[i][1]
            x = np.array([dataPoint[0],dataPoint[1]])
            y = dataLabel
            if (y==0):
                y = -1
            if self.biasState == False:
                if((self.modelWeights.T.dot(x))*y <= 0):
                    delta = 1
                    self.modelWeights = self.modelWeights + x*y
            else:
                if ((self.modelWeights.T.dot(x)+self.bias)*y <= 0):
                    delta = 1
                    self.modelWeights = self.modelWeights + x*y
                    self.bias = self.bias + y
            i+=1
        return delta

    def train(self):
        i= 0
        while i<self.iter:
            delta = self.setDataPoints()
            if delta == 0:
                break
            i+=1
    
    def getParameters(self,i):
        if i == 0:
            return self.modelWeights
        else:
            return self.bias



