#coding=utf-8
'''
计算给定数据集的香农熵
'''
__author__ = 'xiyuanbupt'
from math import log
import operator

def createDataSet():
    dateSet=[[1,1,'yes'],[1,1,'yes'],[1,2,'no'],[0,1,'no'],[0,1,'no']]
    labels=['no surfacing','flippers']
    return dateSet,labels

def calcShannonEnt(dataSet):
    '''
    计算一个数据集的香农熵
    :param dataSet: 数据集，数据集的每一行的自后一条信息为分类信息
    :return:
    '''
    dataCount=len(dataSet)
    labelCount={}
    for data in dataSet:
        label=data[-1]
        labelCount[label]=labelCount.get(label,0)+1
    shannonEnt=0.0
    for label in labelCount:
        prob=float(labelCount[label])/dataCount
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

def splitDataSet(dataSet,axis,value):
    '''
    按照给定的特征划分数据集
    :param dataSet: 数据集
    :param axis: 划分数据集的特征，
    :param value: 需要返回的特征值
    :return:
    '''
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeaturToSplit(dataSet):
    '''
    选择最好的数据集划分方式
    :param dataSet: 数据集
    :return:
    '''
    featurCounts=len(dataSet[0])-1
    lenDataSet=float(len(dataSet))
    baseEnt=calcShannonEnt(dataSet)
    bestIncrement=0.0
    bestFeature=-1
    for i in range(featurCounts):
        featurList=[a[i] for a in dataSet]
        uniqueFeatur=set(featurList)
        newEnt=0.0
        for featur in uniqueFeatur:
            subDataSet=splitDataSet(dataSet,i,featur)
            prob=len(subDataSet)/lenDataSet
            newEnt+=prob*calcShannonEnt(subDataSet)
        newIncrement=baseEnt-newEnt
        if (newIncrement>bestIncrement):
            bestFeature=i
            bestIncrement=newIncrement
    return bestFeature

#递归结束条件，程序遍历完所有划分数据集的属性，或者每个分支下的所有实力都具有相同的分类

def majorityCnt(dataSet):
    '''
    用来判断一个数据集的分类，书中的代码只适用于参数为类列表，本代码适合最后一列为分类的数据
    :param dataSet: 数据集
    :return:
    '''
    classCount={}
    for data in dataSet:
        classCount[data[-1]]=classCount.get(data[-1],0)+1
    sortedClassCount=sorted(classCount.iteritems(),key=
                            operator.itemgetter[1],reverse=True)
    return sortedClassCount[0][0]

#我认为下面创建树的代码为最难理解的一部分，这也是我第一次用Python建树

def createTree(dataSet,labels):
    classList=[data[-1] for data in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeaturToSplit(dataSet)
    label=labels[bestFeat]
    retTree={label:{}}
    del(labels[bestFeat])
    featVals=[exam[bestFeat] for exam in dataSet]
    uniqVals=set(featVals)
    for value in uniqVals:
        sublabels=labels[:]
        retTree[label][value]=createTree(splitDataSet(dataSet,bestFeat,value),sublabels)
    return retTree

def test():
    myDat,labels=createDataSet()
    myTree=createTree(myDat,labels)
    print myTree

if __name__=="__main__":
    test()

