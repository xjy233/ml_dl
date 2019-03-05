import numpy as np
# import tensorflow as tf

from scipy.sparse.linalg import svds
from numpy import *
from numpy import linalg as la

"""
三种距离计算模型
"""
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

def sumSim(inA,inB):
    a = array(inA.T)
    sum = len(a[0])
    return sum

"""
Parameters:
    fileName - 文件名称
Returns:
	re_mat -  观看记录矩阵
	labelMat - 用户名数据集
"""
def loadDataSet(fileName):
    fr = open(fileName,'r',encoding = 'utf-8')
    v_line,labelMat,dataMat = [],[],[];

    for line in fr.readlines():
        lineArr = line.strip().split(',') # 逐行读取，滤除空格等
        if(lineArr[0] == "C"):
            labelMat.append(lineArr[2])   # 添加用户
            dataMat.append(v_line)
            v_line = []
        if(lineArr[0] == "V" ):
            v_line.append(int(lineArr[1]))
    del dataMat[0]

    #填充30000 * 296 的数组，形成稀疏矩阵
    row = len(labelMat)
    re_mat = np.zeros((row, 296))
    i =0
    for list in dataMat:
        for elem in list:
            j = elem - 1000
            re_mat[i,j] = 1
        i += 1

    re_mat = mat(re_mat)
    # print(dataMat)
    #     # print(labelMat)
    #     # print(re_mat)
    return re_mat, labelMat

def loadTestSet(fileName):
    fr = open(fileName, 'r', encoding='utf-8')
    v_line, labelMat, dataMat = [], [], [];

    for line in fr.readlines():
        lineArr = line.strip().split(',')  # 逐行读取，滤除空格等
        if (lineArr[0] == "C"):
            labelMat.append(lineArr[2])  # 添加用户
            dataMat.append(v_line)
            v_line = []
        if (lineArr[0] == "V"):
            v_line.append(int(lineArr[1]))
    del dataMat[0]
    return dataMat, labelMat
"""
不降维的标准评分估计
Parameters:
    dataMat - 数据矩阵
    user - 用户编号
    simMeas - 相似度计算方法
    item - 物品编号
Returns:
    物品的得分
"""
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0: continue
        overLap = nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]  #寻找两个用户都评级的物品
        if len(overLap) == 0: similarity = 0                 #返回的索引值数组是一个2维tuple数组，该tuple数组中包含一维的array数组。其中，一维array向量的个数与a的维数是一致的。
        else: similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j])
        # print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/100

"""
奇异值分解SVD后的评分估计
Parameters:
    dataMat - 数据矩阵
    user - 用户编号
    simMeas - 相似度计算方法
    item - 物品编号
Returns:
	物品的得分
"""
def svdEst(dataMat, user, simMeas, item,xformedItems):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    # U,Sigma,VT = svds(dataMat)    #要用scipy的现代函数处理
    # Sig4 = mat(eye(4)*Sigma[:4]) #arrange Sig4 into a diagonal matrix
    # xformedItems = dataMat.T * U[:,:4] * Sig4.I  #create transformed items
    # print(xformedItems)
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        # print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal

def svd_tran(dataMat):
    U, Sigma, VT = svds(dataMat,70)  # 要用scipy的现代函数处理
    Sig4 = mat(eye(66) * Sigma[:66])  # arrange Sig4 into a diagonal matrix
    xformedItems = dataMat.T * U[:, :66] * Sig4.I  # create transformed items
    return xformedItems
"""
奇异值分解SVD后的评分估计
Parameters:
    dataMat -
    user -
    N=3 -
    simMeas=cosSim -
    estMethod=standEst -
Returns:
	物品的得分最高的前N个物品名
"""
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1]#寻找未评级的物品
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    xformedItems = svd_tran(dataMat)

    for item in unratedItems:
        # estimatedScore = estMethod(dataMat, user, simMeas, item,xformedItems)
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]    #寻找前N个未评级物品


if __name__ == '__main__':
    dataMat, labelMat =loadDataSet('anonymous-msweb.data')
    dataMat_test, labelMat_test = loadTestSet('anonymous-msweb.test')
    # user = 10009
    # user_mat = user-10001
    user_list = [0,1,2,3,4,5,6,7,8,9]
    # result = recommend(dataMat,user_mat, N=10, simMeas=sumSim, estMethod=standEst)
    # print(result)
    i = 0
    for uesr in user_list:
        result = recommend(dataMat, uesr, N=10, simMeas=sumSim, estMethod=standEst)
        print("预测："+str(result))
        print("实际：" + str(dataMat_test[i]))
        i += 1




