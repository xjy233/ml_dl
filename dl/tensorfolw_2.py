import tensorflow as tf
from numpy.random import RandomState

batch_size = 8
w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_= tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# y = tf.sigmoid(y)
# cross_entropy = tf.reduce_mean(tf.square(y_-y))
# train_step=tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                                + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)
X = rdm.rand(128,2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 输出目前（未经训练）的参数取值。
    # print(sess.run(w1))
    # print(sess.run(w2))
    # print("\n")

    # 训练模型。
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % 128
        end = (i * batch_size) % 128 + batch_size
        sess.run([train_step, y, y_], feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

    # 输出训练后的参数取值。
    # print("\n")
    print("X:\n", X)
    # print(sess.run(w1))
    # print("a:\n", sess.run(a, feed_dict={x: X}))
    # print(sess.run(w2))
    # print(Y)
    # print("y:\n", sess.run(y, feed_dict={x: X}))
    yyu = sess.run(y, feed_dict={x: X})
    for i in range(len(yyu)):
        print("y:{}  Y:{}\n".format(yyu[i],Y[i]))