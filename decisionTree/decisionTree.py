#coding=utf-8
'''
计算给定数据集的香农熵
'''
__author__ = 'xiyuanbupt'
from math import log

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
        prob=float(labelCount[label])/dataSet
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

