# # 总结：
# # 训练神经网络的过程可以分为以下三个步骤：
# # 1、定义神经网络的结构和前向传播的输出结果
# # 2、定义损失函数以及选择反向传播优化的算法
# # 3、生成会话（tf.Session）并且在训练数据上反复运行反向传播优化算法
# # coding:utf-8
# import tensorflow as tf
# from numpy.random import RandomState
#
# # BATCH_SIZE这是定义的一个数量，即一次训练模型，投入的样例数，按理说应该是一次投入所有训练数据的
# # 但是实际操作中一次投入太多数据，会导致电脑死机，因此只好对投入样例数，进行限制
# BATCH_SIZE = 10
# # 生成模拟数据
# # randomState 函数中数字1,相当于一个seed种子，每次产生的随机数都是相同的
# rdm = RandomState(1)
# # 产生一个128行×2列的随机矩阵
# datasize = 80
# X = rdm.rand(datasize, 2)
# # 产生一个布尔型结果矩阵128×1
# # Y = [[int(x1 + x2) < 1] for (x1, x2) in X]
#
# Y = [[int(x1+x2 > 1)] for (x1, x2) in X]
#
# # 定义神经网络的常量，参数，输入节点，输出节点，以及前向传播过程
# # Variable定义一个变量，需要调用初始化，才能产生值
# # random_normal 产生一个正太随机矩阵，shape=[2,3],stddev是正太分布的标准差，seed随机数种子，设置后每次产生的数字都相同
# w1 = tf.Variable(tf.random_normal([2, 100], stddev=1, seed=1))
# w2 = tf.Variable(tf.random_normal([100, 1], stddev=1, seed=1))
#
# # placeholder占位符，执行时，在通过feed_dict将值传入，dtype：数据类型，shape：数据形状，name：名称
# # 如果在定义占位符时，不能确定值的形状时，用None表示
# x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
# y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")
#
# # 前向传播的过程
# # matmul就是矩阵相乘
# a = tf.matmul(x, w1)
# y = tf.matmul(a, w2)
# y = tf.sigmoid(y)
#
# # 定义损失函数及反向传播算法
# # cross_entropy是交叉熵
# # tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max
# # cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
# # cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
#                                 # + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
# cross_entropy = tf.reduce_mean(tf.square(y_-y))
# # train_step 就是每次可训练出一组w就是一次反向传播
# # 下面给出里三种反向传播的算法传入的是学习率
# # train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
# train_step=tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
# # train_step=tf.train.MomentumOptimizer(0.001,0.9).minimize(cross_entropy)
#
# # 创建一个会话来运行Tensorflow程序，反复运行反向传播
# # tf中运行必须放在session对象中，且运行过后，要关闭session
# with tf.Session()as sess:
#     # 初始化变量，也就是上面的Variable变量
#     init_op = tf.initialize_all_variables()
#     sess.run(init_op)
#     # 输出目前（未经训练）的参数取值
#     # 这里为何还要使用sess.run(w1)来进行操作？
#     # 因为此时w1还是个变量，是个对象，直接打印出来是对象的地址
#     print("w1:\n", sess.run(w1))
#     print("w2:\n", sess.run(w2))
#
#     # 训练模型
#     # 训练步数
#     STEPS = 10000
#     for i in range(STEPS):
#         start = (i * BATCH_SIZE) % datasize
#         end = (i * BATCH_SIZE) % datasize + BATCH_SIZE
#         sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
#         if i % 1000 == 0:
#             total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
#             print("After %d training steps,cross entropy on all data is %g" % (i, total_cross_entropy))
#     # 输出训练后的参数取值。
#     # print("\n")
#     # print("X:\n", X)
#     print(sess.run(w1))
#     # print("a:\n", sess.run(a, feed_dict={x: X}))
#     print(sess.run(w2))
#     # print(Y)
#     # yyu = sess.run(y, feed_dict={x: X})
#     # for i in range(len(yyu)):
#     #     print("y:{}  Y:{}\n".format(yyu[i],Y[i]))


