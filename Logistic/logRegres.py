#coding=utf-8
__author__="xiyuanbupt"
from numpy  import *
def loadDataSet():
    dataMat=[];labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn,classLables):
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLables).transpose()
    m,n=dataMatrix.shape
    alpha=0.001
    maxCycles=500
    weights=ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=labelMat-h
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights

def stocGradAscent0(dataMatrix,classLabels,times=1):
    '''
    随机梯度上升算法，times代表迭代的次数
    :param dataMatrix:
    :param classLabels:
    :param time:
    :return:
    '''
    m,n=shape(dataMatrix)
    alpha=0.01
    weights=ones(n)
    for k in range(times):
        for i in range(m):
            h=sigmoid(sum(dataMatrix[i]*weights))
            error=classLabels[i]-h
            weights=weights+alpha*error*array(dataMatrix[i])
    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=(dataArr.shape)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i]==1):
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=array((-weights[0]-weights[1]*x)/weights[2])
    print len(x),len(y)
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    return 1.0 if prob>0.5 else 0.0

if __name__=="__main__":
    a,b=loadDataSet()
    weights=stocGradAscent0(a,b)
    plotBestFit(weights)