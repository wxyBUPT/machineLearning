#coding=utf-8
__author__='xiyuanbupt'
from NBM import createVocabList,bagOfWords2VecMN,trainNB,classifyNB
import random

def textParse(bigString):
    '''
    将长文本切分为单词列表
    :param bigString: 长文本
    :return:
    '''
    import re
    listOfTokens=re.split(r'/W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

class DataSourceFrom(object):
    '''
    获得用于训练和用于测试的数据集
    '''
    def getData(self):
        '''
        用于在子类中重写的函数，提供从不同数据源中获得数据的功能
        :return:
        '''
        trainList=[]
        trainClassifyVec=[]
        testList=[]
        testClassifyVec=[]
        return trainList,trainClassifyVec,testList,testClassifyVec
    def __init__(self):
        self.meaningDict={}
        trainList,trainClassifyVec,self.testList,self.testClassifyVec=self.getData()
        vocabList=createVocabList(trainList)
        trainMat=[]
        for postingData in trainList:
            trainMat.append(bagOfWords2VecMN(vocabList,postingData))
        self.p0V,self.p1V,self.pAb=trainNB(trainMat,trainClassifyVec)
        self.vocabList=vocabList

    def classifyNB(self,listWord2classify):
        '''
        对给定的词组列表进行分类
        :param vec2classify:
        :return:
        '''
        vectorWord2classify=bagOfWords2VecMN(self.vocabList,listWord2classify)
        res=classifyNB(vectorWord2classify,self.p0V,self.p1V,self.pAb)
        return res

    def getAccurayRate(self):
        numOfTest=len(self.testClassifyVec)
        numOfSuc=0
        for i,testData in enumerate(self.testList):
            res=self.classifyNB(bagOfWords2VecMN(self.vocabList,testData))
            if res==self.testClassifyVec[i]:
                numOfSuc+=1
        if numOfTest!=0:
            return float(numOfSuc/numOfTest)
        else:
            return 0

class DataSourceFromFile(DataSourceFrom):

    def getData(self):
        trainList=[]
        trainClassifyVec=[]
        testList=[]
        testClassifyVec=[]
        for i in range(1,26):
            wordList=textParse(open('email/spam/%d.txt'%(i)).read())
            trainList.append(wordList)
            trainClassifyVec.append(1)
            wordList=textParse(open('email/ham/%d.txt'%(i)).read())
            trainList.append(wordList)
            trainClassifyVec.append(0)
        for i in range(10):
            randIndex=int(random.uniform(0,len(trainList)))
            testList.append(trainList[randIndex])
            testClassifyVec.append(trainClassifyVec[randIndex])
            del(trainList[randIndex])
            del(trainClassifyVec[randIndex])
        return trainList,trainClassifyVec,testList,testClassifyVec

if __name__=="__main__":
    a=DataSourceFromFile()
    print a.getAccurayRate()