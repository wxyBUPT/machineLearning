#coding=utf-8
__author__ = 'xiyuanbupt'
import numpy
import operator

class DataSetSource(object):

    def __init__(self):
        pass

    def getDataSet(self):
        group=numpy.array([[1.0,1.1],[1.0,1.1],[0,0],[0,0.1]])
        labels=['A','A','B','B']
        return group,labels

    def classify(self,inX,dataSet,labels,k):
        '''
        k-近邻算法
        :param inX: 用于分类的向量
        :param dataSet: 用于训练样本集的dataSet
        :param labels: 标签向量
        :param k: 用于选择最近邻居的数码
        :return:
        '''
        dataSetSize=dataSet.shape[0]

        #距离计算
        diffMat=numpy.tile(inX,(dataSetSize,1))-dataSet
        sqDiffMat=diffMat**2
        sqDistances=sqDiffMat.sum(axis=1)
        distances=sqDistances**0.5
        sortedDistIndicies=distances.argsort()

        #获得距离最近的qiank个的每一个标签+1
        classCount={}
        for i in range(k):
            label=labels[sortedDistIndicies[i]]
            classCount[label]=classCount.get(label,0)+1

        sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
        return sortedClassCount[0][0]

    def file2matrix(self,fileName,attriCount):
        '''
        将文本记录转化为NumPy的解析程序
        :param fileName: 文件路径
        :param attriCount: 每一个训练数据项的维度
        :return:
        '''
        fr=open(fileName)
        lines=fr.readlines()
        numOfLines=len(lines)
        returnMat=numpy.zeros((numOfLines,attriCount))
        labels=[]
        index=0
        for line in lines:
            line=line.strip()
            line=line.split('\t')
            returnMat[index,:]=line[0:attriCount]
            labels.append(line[-1])
            index+=1
        return returnMat,labels

    def autoNorm(self,dataSet):
        '''
        归一化特征值
        :param dataSet: numpy.nndarray 的训练矩阵
        :return:归一化的训练矩阵，每列的范围数组，每列的最小值
        '''
        minVals=dataSet.min(0)
        maxVals=dataSet.max(0)
        ranges=maxVals-minVals
        normDataSet=numpy.zeros((numpy.shape(dataSet)))
        m=dataSet.shape[0]
        normDataSet=dataSet-numpy.tile(minVals,(m,1))
        normDataSet=normDataSet/numpy.tile(ranges,(m,1))
        return normDataSet,ranges,minVals

    def testShow(self,mat,labels):
        import matplotlib
        import matplotlib.pylab
        fig=matplotlib.pylab.figure()
        ax=fig.add_subplot(111)
        ax.scatter(mat[:,1],mat[:,2])
        matplotlib.pylab.show()

if __name__=="__main__":
    dataSetSource=DataSetSource()
    dataSet,labels=dataSetSource.file2matrix('/Users/xiyuanbupt/PycharmProjects/machineLearning/machinelearninginaction/Ch02/datingTestSet.txt',3)
    dataSet,y,z=dataSetSource.autoNorm(dataSet)
    print len(labels)
    print len(dataSet)
    print type(dataSet)
    print dataSet.shape
    dataSetSource.testShow(dataSet,labels)