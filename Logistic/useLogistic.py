#coding=utf-8
'''
用Logistic 从疝气病症预测病马的死亡率
'''
__author__="xiyuanbupt"
from logRegres import *
from numpy import *

def useLogistic():
    frTrain=open('horseColicTraining.txt')
    listData=[]
    listLabels=[]
    for line in frTrain.readlines():
        line=line.strip().split('\t')
        lineData=[]
        for i in range(21):
            lineData.append(float(line[i]))
        listData.append(lineData)
        listLabels.append(float(line[21]))
    weights=stocGradAscent0(array(listData),listLabels,200)

    frTest=open('horseColicTest.txt')
    numTestVec=0.0
    numErrorVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        line=line.strip().split('\t')
        lineData=[]
        for i in range(21):
            lineData.append(float(line[i]))
        if int(classifyVector(lineData,weights))!=int(line[21]):
            numErrorVec+=1.0
    errorRate=numErrorVec/numTestVec
    print u'错误率是 %f'%(errorRate)




if __name__=="__main__":
    useLogistic()