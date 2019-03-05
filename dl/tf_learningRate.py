import tensorflow as tf

TRAINING_STEPS = 10
LEARNING_RATE = 0.1
x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
y = tf.square(x)
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        sess.run(train_op)
        x_value = sess.run(x)
        print("After %s iteration(s): x%s is %f." % (i + 1, i + 1, x_value))

train_op1 = tf.train.GradientDescentOptimizer(0.05).minimize(y)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        sess.run(train_op1)
        if i % 10 == 0:
            x_value = sess.run(x)
            print("After %s iteration(s): x%s is %f." % (i + 1, i + 1, x_value))

#使用指数学习率
global_step = tf.Variable(0)
LEARNING_RATE1 = tf.train.exponential_decay(0.2, global_step, 1, 0.85, staircase=True)
train_op2 = tf.train.GradientDescentOptimizer(LEARNING_RATE1).minimize(y, global_step=global_step)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(70):
        sess.run(train_op2)
        if i % 5 == 0:
            LEARNING_RATE_value1 = sess.run(LEARNING_RATE1)
            x_value = sess.run(x)
            print("After %s iteration(s): x%s is %f, learning rate is %f." % (
            i + 1, i + 1, x_value, LEARNING_RATE_value1))