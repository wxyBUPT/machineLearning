#coding=utf-8
__author__ = 'xiyuanbupt'
from NBM import loadDataSet,createVocabList,setOfWords2Vec,trainNB

def testTrainNB():
    listOPosts,trainCategory=loadDataSet()
    vocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(vocabList,postinDoc))
    p0V,p1V,pAb=trainNB(trainMat,trainCategory)

if __name__=="__main__":
    testTrainNB()
