#coding=utf-8
__author__ = 'xiyuanbupt'
import numpy
from math import log
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    '''
    创建数据集中的词汇表
    :param dataSet: 数据集
    :return:
    '''
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet | set(document)  #取并集
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    '''
    获得文档向量
    :param vecabList: 词汇集
    :param inputSet: 输入的文档
    :return:
    '''
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print u'%s not in my Vocabulary!'%word
    return returnVec

def trainNB(trainMatrix,trainCategory):
    '''
    朴素贝叶斯分类器训练函数
    :param trainMatrix:
    :param trainCategory:
    :return:
    '''
    trainDocsCount=len(trainMatrix)
    wordsCount=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(trainDocsCount)
    p0Num=numpy.ones(wordsCount)
    p1Num=numpy.ones(wordsCount)
    p0Denom=2.0
    p1Denom=2.0
    for i in range(trainDocsCount):
        if trainCategory[i]==0:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
            pass
        else :
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
            pass
    p0Vect=p0Num/p0Denom
    p1Vect=p1Num/p1Denom
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    '''
    朴素贝叶斯分类函数
    :param vec2Classify:
    :param p0Vec:
    :param p1Vec:
    :param pClass1:
    :return:
    '''
    p1=sum( vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def bagOfWords2VecMN(vocabList,inputSet):
    '''
    词袋模型
    :param vocabList: 单词集
    :param inputSet: 输入的文本
    :return:
    '''
    retVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            retVec[vocabList.index(word)]+=1
    return retVec
