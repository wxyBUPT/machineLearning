#coding=utf-8
'''
本文件使用matplotlib画图
'''
__author__ = 'xiyuanbupt'
import matplotlib.pyplot as plt
#定义文本框和箭头的格式
decisionNode=dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")

def createPlot():
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1=plt.subplot(111,frameon=False)
    plotNode(u'决策节点',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode(u'叶节点',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()

def plotNode(nodeText,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeText,xy=parentPt,
                            xycoords='axes fraction',
                            xytext=centerPt,textcoords='axes fraction',
                            va='center',ha='center',bbox=nodeType,
                            arrowprops=arrow_args)

if __name__=="__main__":
    createPlot()