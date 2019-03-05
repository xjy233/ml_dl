'''
k Means Clustering for Finding Similar Time Series in Sales Transaction Data
'''
from numpy import *
import pandas as pd
from pandas import DataFrame,Series

#加载数据
def loadDataSet(fileName):  # general function to parse tab -delimited floats
    fr = pd.read_csv(fileName)
    #切分数据
    examDf = DataFrame(fr)
    new_examDf1 = examDf.ix[:, :1]
    new_examDf2 = examDf.ix[:, 55:]
    result = new_examDf1.join(new_examDf2)
    dataset = result.ix[ :, 1:].values

    return dataset

#计算距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # la.norm(vecA-vecB)

#初始化k个点
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))  # create centroid mat
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids

#普通Kmeans
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))  # create mat to assign data points
    # to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    i = 0
    while (clusterChanged or i < 100):
        clusterChanged = False
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = inf;
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(centroids)
        for cent in range(k):  # recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean
        i += 1
    return centroids, clusterAssment

#二分K-均值算法
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]  # create a list with one centroid
    for j in range(m):  # calc initial Error
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0],
                               :]  # get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])  # compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # replace a centroid with two best centroids
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0],
        :] = bestClustAss  # reassign new clusters, and SSE
    return mat(centList), clusterAssment

#测试kmeans
def test_Kmeans():
    date = loadDataSet('Sales_Transactions_Dataset_Weekly.csv')
    #v为每个簇包含的产品数，d为总共产品数，k为总共的簇的个数
    v = 4;d = len(date)
    k = int(d/v)
    centroids, clusterAssment = kMeans(date, k,distEclud,randCent)

    file = open('cluster.txt', 'w', encoding='utf-8')
    #写入数据到txt文本
    for i in range(k):
        file.write("第{}组：".format(i)+"\n")
        for j in range(len(clusterAssment)):
            if(clusterAssment[j,0] == i):
                file.write("产品名称：P{}   距离为：{:.4f}".format(j,clusterAssment[i,1]) +"\n")


if __name__ == '__main__':
    test_Kmeans()


