#coding=utf-8
__author__ = 'xiyuanbupt'
from numpy import *
from NBM import loadDataSet,createVocabList,setOfWords2Vec,trainNB,classifyNB

def testTrainNB():
    listOPosts,trainCategory=loadDataSet()
    vocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(vocabList,postinDoc))
    p0V,p1V,pAb=trainNB(trainMat,trainCategory)
    testEntry=['love','wang','xi','my','dalmation']
    vec2classify=array(setOfWords2Vec(vocabList,testEntry))
    print testEntry,'classify as:',classifyNB(vec2classify,p0V,p1V,pAb)



if __name__=="__main__":
    testTrainNB()
