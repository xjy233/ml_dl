# -*- coding: UTF-8 -*-

from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import operator
import collections

'''
函数说明:打开并解析文件，对数据进行分类

Parameters:
	filename - 文件名
Returns:
	returnMat - 特征矩阵
	classLabelVector - 分类Label向量

Modify:
	2017-03-24
'''
def file2matrix(filename):
#打开文件,此次应指定编码
	fr = open(filename,'r',encoding = 'utf-8')
#读取文件所有内容
	arrayOLines = fr.readlines()
#针对有BOM的UTF-8文本，应该去掉BOM，否则后面会引发错误。
	arrayOLines[0]=arrayOLines[0].lstrip('\ufeff')
#得到文件行数
	numberOfLines = len(arrayOLines)
#返回的NumPy矩阵,解析完成的数据:numberOfLines行,16列
	returnMat = np.zeros((numberOfLines,16))
#返回的分类标签向量
	classLabelVector = []
#行的索引值
	index = 0
	for line in arrayOLines:
#s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
		line = line.strip()
#使用s.split(str="",num=string,cout(str))将字符串根据','分隔符进行切片。
		listFromLine = line.split(',')
#将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
		returnMat[index,:] = listFromLine[1:17]
#类别
		classLabelVector.append(listFromLine[0])
		index += 1
	return returnMat, classLabelVector

"""
函数说明:对数据进行归一化

Parameters:
	dataSet - 特征矩阵
Returns:
	normDataSet - 归一化后的特征矩阵
	ranges - 数据范围
	minVals - 数据最小值

Modify:
	2017-03-24
"""
def autoNorm(dataSet):
	#获得数据的最小值
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	#最大值和最小值的范围
	ranges = maxVals - minVals
	#shape(dataSet)返回dataSet的矩阵行列数
	normDataSet = np.zeros(np.shape(dataSet))
	#返回dataSet的行数
	m = dataSet.shape[0]
	#原始值减去最小值
	normDataSet = dataSet - np.tile(minVals, (m, 1))
	#除以最大和最小值的差,得到归一化数据
	normDataSet = normDataSet / np.tile(ranges, (m, 1))
	#返回归一化数据结果,数据范围,最小值
	return normDataSet, ranges, minVals

"""
函数说明:kNN算法,分类器

Parameters:
	inX - 用于分类的数据(测试集)
	dataSet - 用于训练的数据(训练集)
	labes - 分类标签
	k - kNN算法参数,选择距离最小的k个点
Returns:
	sortedClassCount[0][0] - 分类结果

Modify:
	2017-11-09 by Cugtyt 
		* GitHub(https://github.com/Cugtyt) 
		* Email(cugtyt@qq.com)
		Use list comprehension and Counter to simplify code
	2017-07-13
"""
def classify0(inx, dataset, labels, k):
	# 计算距离
	dist = np.sum((inx - dataset)**2, axis=1)**0.5
	# k个最近的标签
	k_labels = [labels[index] for index in dist.argsort()[0 : k]]
	# 出现次数最多的标签即为最终类别
	label = collections.Counter(k_labels).most_common(1)[0][0]
	return label

"""
函数说明:分类器测试函数

Parameters:
	无
Returns:
	normDataSet - 归一化后的特征矩阵
	ranges - 数据范围
	minVals - 数据最小值

Modify:
	2017-03-24
"""
def datingClassTest():
	#打开的文件名
	filename = 'letter-recognition.data'
	#将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
	datingDataMat, datingLabels = file2matrix(filename)
	#取所有数据的百分之十
	hoRatio = 0.10
	#数据归一化,返回归一化后的矩阵,数据范围,数据最小值
	normMat, ranges, minVals = autoNorm(datingDataMat)
	#获得normMat的行数
	m = normMat.shape[0]
	#百分之十的测试数据的个数
	numTestVecs = int(m * hoRatio)
	#分类错误计数
	errorCount = 0.0

	for i in range(numTestVecs):
		#前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],
			datingLabels[numTestVecs:m], 4)
		print("分类结果:%s\t真实类别:%s" % (classifierResult, datingLabels[i]))
		if classifierResult != datingLabels[i]:
			errorCount += 1.0
	print("错误率:%f%%" %(errorCount/float(numTestVecs)*100))

"""
函数说明:main函数

Parameters:
	无
Returns:
	无

Modify:
	2017-03-24
"""
if __name__ == '__main__':
	datingClassTest()
