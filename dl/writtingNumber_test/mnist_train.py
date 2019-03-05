import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import os
from numpy import *

#定义神经网络结构相关的参数。
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="MNIST_model/"
MODEL_NAME="mnist_model"

#定义训练过程。
def train(xs,ys):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(      #tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=True/False)
        LEARNING_RATE_BASE,                             #每decay_steps轮衰减decay_rate
        global_step,
        BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            start = (i * BATCH_SIZE) % 1500
            end = (i * BATCH_SIZE) % 1500 + BATCH_SIZE
            # xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _,loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs[start:end], y_: ys[start:end]})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

#主程序入口。
def main(argv=None):
    #input_data.read_data_sets函数生成的类会自动将MNIST数据集划分为train=55000、validation=5000和test=10000三个数据集。
    # mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    xs,ys,txs,tys= load_date()
    train(xs,ys)

"""
处理数据
"""
def load_date():
    input_xs = []
    input_ys = []
    percent = 0.1
    lines = open("semeion.data").readlines()
    for line in lines:
        list = line.strip().split(' ')
        input_x = list[:-10]
        input_y = list[-10:]
        input_xs.append(input_x)
        input_ys.append(input_y)

    num = int(percent * len(input_xs))
    train_x = mat(input_xs[:-num])
    train_y = mat(input_ys[:-num])
    test_x = mat(input_xs[-num:])
    test_y = mat(input_ys[-num:])
    return train_x,train_y,test_x,test_y

if __name__ == '__main__':
    tf.app.run()