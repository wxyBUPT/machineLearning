#coding=utf-8
'''
使用kNN来解决书中的实际问题
'''
__author__ = 'xiyuanbupt'
from kNN import DataSetSource
import numpy

def classifyPerson():
    a=float(raw_input(u"输入a"))
    b=float(raw_input(u'输入b'))
    c=float(raw_input(u'输入c'))
    dataSetSource=DataSetSource()
    mat,labels=dataSetSource.file2matrix("datingTestSet.txt",3)
    inX=numpy.array([a,b,c])
    mat,range,minV=dataSetSource.autoNorm(mat)
    label=dataSetSource.classify(inX,mat,labels,4)
    print label

if __name__=="__main__":
    classifyPerson()