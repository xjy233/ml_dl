import numpy as np

'''
函数说明:打开并解析文件，对数据进行分类
Returns:
    retrianList - 训练集
    retesttrain - 测试集
    classVec - 类别
'''
def loadDataSet():
#打开文件,此次应指定编码
    filename = 'nursery.data'
    fr = open(filename,'r',encoding = 'utf-8')
#读取文件所有内容
    arrayOLines = fr.readlines()
#针对有BOM的UTF-8文本，应该去掉BOM，否则后面会引发错误。
    arrayOLines[0]=arrayOLines[0].lstrip('\ufeff')
#返回的分类标签向量
    classVec = [];postingList = [];
    for line in arrayOLines:
#s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        listFromLine = line.split(',')
#将数据前8列提取出来
        postingList.append(listFromLine[0:8])
#类别
        classVec.append(listFromLine[-1])


    m = len(postingList)
    a = int(0.1 * m)
    x = m-a
    retrianList = postingList[:x]
    retesttrain = postingList[x:]

    return retrianList,retesttrain, classVec


"""
函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0

Parameters:
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量,词集模型

"""
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)                                    #创建一个其中所含元素都为0的向量
    for word in inputSet:                                                #遍历每个词条
        if word in vocabList:                                            #如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec                                                    #返回文档向量

"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表//去重合并

Parameters:
    dataSet - 整理的样本数据集
Returns:
    vocabSet - 返回不重复的词条列表，也就是词汇表

"""
def createVocabList(dataSet):
    vocabSet = set([])                      #创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document) #取并集
    return list(vocabSet)

"""
函数说明:朴素贝叶斯分类器训练函数

Parameters:
    trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
    p0Vect,p1Vect,p2Vect,p3Vect,p4Vect    词向量顺序与下面的类概率一一对应
    Pnot_recom,Precommend,Pvery_recom,Ppriority,Pspec_prior
"""
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)                            #计算训练的文档数目
    numWords = len(trainMatrix[0])                            #计算每篇文档的词条数

    num_not_recom = 0;num_recommend = 0;num_very_recom = 0;num_priority = 0;num_spec_prior = 0;
    for word in trainCategory:
        if word == "not_recom":
            num_not_recom += 1
        elif word == "recommend":
            num_recommend += 1
        elif word == "very_recom":
            num_very_recom += 1
        elif word == "priority":
            num_priority += 1
        elif word == "spec_prior":
            num_spec_prior += 1

    # 文档属于某类的概率
    Pnot_recom = num_not_recom/float(numTrainDocs)
    Precommend = num_recommend / float(numTrainDocs)
    Pvery_recom = num_very_recom / float(numTrainDocs)
    Ppriority = num_priority / float(numTrainDocs)
    Pspec_prior = num_spec_prior / float(numTrainDocs)

    # 创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑,注意p0Num-p4Num和上面类对应
    p0Num = np.ones(numWords);p1Num = np.ones(numWords);p2Num = np.ones(numWords)
    p3Num = np.ones(numWords);p4Num = np.ones(numWords)
    p0Denom = 2.0;p1Denom = 2.0;p2Denom = 2.0;p3Denom = 2.0;p4Denom = 2.0  # 分母初始化为2,拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == "not_recom":
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
        elif trainCategory[i] == "recommend":
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        elif trainCategory[i] == "very_recom":
            p2Num += trainMatrix[i]
            p2Denom += sum(trainMatrix[i])
        elif trainCategory[i] == "priority":
            p3Num += trainMatrix[i]
            p3Denom += sum(trainMatrix[i])
        elif trainCategory[i] == "spec_prior":
            p4Num += trainMatrix[i]
            p4Denom += sum(trainMatrix[i])
    p0Vect = np.log(p0Num / p0Denom);p1Vect = np.log(p1Num / p1Denom);  # 取对数，防止下溢出
    p2Vect = np.log(p2Num / p2Denom);p3Vect = np.log(p3Num / p3Denom);p4Vect = np.log(p4Num / p4Denom)
    return p0Vect,p1Vect,p2Vect,p3Vect,p4Vect,Pnot_recom,Precommend,Pvery_recom,Ppriority,Pspec_prior

"""
函数说明:朴素贝叶斯分类器分类函数
Parameters:
    vec2Classify - 待分类的词条数组
    p0Vect,p1Vect,p2Vect,p3Vect,p4Vect - 对应下面类的概率词向量
    Pnot_recom,Precommend,Pvery_recom,Ppriorityspec_prior,Pspec_prior - 类的概率
Returns:
    not_recom，recommend，very_recom，priority，spec_prior
"""
def classifyNB(vec2Classify,p0Vect,p1Vect,p2Vect,p3Vect,p4Vect,Pnot_recom,Precommend,Pvery_recom,Ppriority,Pspec_prior):
    #对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vect) + np.log(Pnot_recom)
    p1 = sum(vec2Classify * p1Vect) + np.log(Precommend)
    p2 = sum(vec2Classify * p2Vect) + np.log(Pvery_recom)
    p3 = sum(vec2Classify * p3Vect) + np.log(Ppriority)
    p4 = sum(vec2Classify * p4Vect) + np.log(Pspec_prior)
    p = max(p0,p1,p2,p3,p4)
    if p == p0:
        return "not_recom"
    elif p == p1:
        return "recommend"
    elif p == p2:
        return "very_recom"
    elif p == p3:
        return "priority"
    elif p == p4:
        return "spec_prior"

"""
函数说明:测试朴素贝叶斯分类器
Parameters:无
Returns:无
"""
def testingNB():
    listOPosts,testEntry,listClasses = loadDataSet()									#创建实验样本
    myVocabList = createVocabList(listOPosts)								#创建词汇表
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))				#将实验样本向量化

    p0Vect, p1Vect, p2Vect, p3Vect, p4Vect,\
    Pnot_recom, Precommend, Pvery_recom, Ppriorityspec_prior, Pspec_prior = trainNB0(np.array(trainMat),np.array(listClasses))		#训练朴素贝叶斯分类器								#测试样本1

    # 分类错误计数
    errorCount = 0.0
    i=len(listOPosts)
    for text in testEntry:
        thisDoc = np.array(setOfWords2Vec(myVocabList, text))
        classifierResult = classifyNB(thisDoc,p0Vect,p1Vect,p2Vect,p3Vect,p4Vect,
                                      Pnot_recom,Precommend,Pvery_recom,Ppriorityspec_prior,Pspec_prior)
        print("分类结果:%s\t真实类别:%s" % (classifierResult, listClasses[i]))
        if classifierResult != listClasses[i]:
            errorCount += 1.0
        i += 1
    print("错误率:%f%%" % (errorCount / float(int(len(testEntry))) * 100))

if __name__ == '__main__':
    testingNB()