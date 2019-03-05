import pandas as pd
from pandas import DataFrame,Series
from numpy import *

'''
函数说明:打开并解析文件，对数据进行分类
Returns:
	X_train - 训练特征
	X_test - 测试特征
	Y_train - 训练类别即rating
	Y_test - 测试类别即rating
'''
def loadDataSet():
    datafile = u'E:\\研究生科目相关\\数据分析工具实践\\第三次\\第3次作业\\回归算法\\2014 and 2015 CSM dataset.xlsx'#文件所在位置，u为防止路径中有中文名称，此处没有，可以省略
    # datafile是excel文件，所以用read_excel,如果是csv文件则用read_csv
    data = pd.read_excel(datafile)
    # data.drop(['Gross','Genre','Budget','Screens','Sequel'],axis=1,inplace=True)
    data.drop(['Gross','Sentiment','Views','Likes','Dislikes','Comments','Aggregate Followers'],
              axis=1,inplace=True)
    # del data['Gross']

    data=data.fillna(1.0)
    examDf = DataFrame(data)

    #数据清洗,比如第1.2列有可能是日期，这样的话我们就只需要从第3列开始的数据，
    new_examDf = examDf.ix[:,2:]

    test_size = 0.1*len(new_examDf)

    X_train = new_examDf.ix[test_size:,1:].values
    X_test = new_examDf.ix[:test_size,1:].values
    Y_train = new_examDf.ix[test_size:,0].values
    Y_test = new_examDf.ix[:test_size,0].values

    return X_train,X_test,Y_train,Y_test

'''
函数说明:线性回归lwlr
Returns:
	testPoint * ws - 单行预测结果
'''
def lwlr(testPoint,xArr,yArr,k):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))

    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)

    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))

    return testPoint * ws

'''
函数说明:遍历所有测试集数据，并将结果返回
Returns:
	yHat - 预测结果
'''
def lwlrTest(testArr,xArr,yArr,k):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

'''
函数说明:测试统计
'''
def testingNB():
    X_train, X_test, Y_train, Y_test = loadDataSet()

    Y_pred = lwlrTest(X_test,X_train,Y_train,3e+6)       #统计k的取值contradional = 3e+6 25%  social = 4.8e+5 16.6%
    errorCount = 0
    for i in range(len(Y_pred)):
        print("第%d次：预测：%s\t实际：%s" % (i,Y_pred[i], Y_test[i]))
        if abs(Y_pred[i] - Y_test[i]) > 1.0:
            errorCount += 1
    print("错误率:%f%%" % (errorCount / float(len(Y_pred)) * 100))

if __name__ == '__main__':
    testingNB()

