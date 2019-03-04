# -- encoding:utf-8 --
'''
用类似lenet网络识别mnist数据
'''
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# # 屏蔽tf中的log信息
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 50
display_step = 1
# 加载数据
mnist = input_data.read_data_sets('E:\deep_learning\data\mnist', one_hot=True)

# 数据分为3类，train，test，和validation
x_train = mnist.train.images       # (55000, 784)
y_train = mnist.train.labels       # (55000, 10)
x_test = mnist.test.images         # (10000, 784)
y_test = mnist.test.labels         # (10000, 10)
x_valid = mnist.validation.images  # (5000, 784)
y_valid = mnist.validation.labels  # (5000, 10)

train_sample_number = mnist.train.num_examples
train_num = train_sample_number // batch_size

# build model
x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')
keep_prob = tf.placeholder(tf.float32)
# 转化为图像
image = tf.reshape(x, shape=[-1, 28, 28, 1])
# 第一个卷积层
# 定义filter
w1 = tf.Variable(name='w1', initial_value=tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
# 定义bias
b1 = tf.Variable(name='b1', initial_value=tf.constant(value=0.1, dtype=tf.float32, shape=[32]))
# 卷积+bias Relu激活函数
conv1 = tf.nn.relu(tf.nn.conv2d(image, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
# 池化
pool1 = tf.nn.max_pool(value=conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第二个卷积+池化
w2 = tf.Variable(name='w2', initial_value=tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b2 = tf.Variable(name='b2', initial_value=tf.constant(value=0.1, dtype=tf.float32, shape=[64]))
conv2 = tf.nn.relu(tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
pool2 = tf.nn.max_pool(value=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 全连接层1
w3 = tf.Variable(name='w3', initial_value=tf.truncated_normal([7*7*64, 1024], stddev=0.1))
b3 = tf.Variable(name='b3', initial_value=tf.constant(value=0.1, dtype=tf.float32, shape=[1024]))
fc1_input = tf.reshape(pool2, shape=[-1, 7*7*64])
fc1 = tf.nn.relu(tf.matmul(fc1_input, w3) + b3)

# 全连接层2
w4 = tf.Variable(name='w4', initial_value=tf.truncated_normal([1024, 84], stddev=0.1))
b4 = tf.Variable(name='b4', initial_value=tf.constant(value=0.1, dtype=tf.float32, shape=[84]))
fc2 = tf.nn.relu(tf.matmul(fc1, w4) + b4)

# 全连接层3
w5 = tf.Variable(name='w5', initial_value=tf.truncated_normal([84, 10], stddev=0.1))
b5 = tf.Variable(name='b5', initial_value=tf.constant(value=0.1, dtype=tf.float32, shape=[10]))
y_ = tf.nn.softmax(tf.matmul(fc2, w5) + b5)

# loss函数
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), reduction_indices=[1]))
# 优化Adam
train = tf.train.AdamOptimizer(0.0001).minimize(loss)
# accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))

# 模型初始化
init = tf.global_variables_initializer()

# 开始训练
with tf.Session() as sess:
    # 变量初始化
    sess.run(init)
    # 模型保存
    saver = tf.train.Saver()
    # 开始训练
    epoch = 0
    while 1:
        # 分批次训练
        avg_loss = 0
        for i in range(train_num):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            feed_dict = {x:batch_x, y:batch_y}
            # 训练
            sess.run(train, feed_dict=feed_dict)
            # 获取loss
            avg_loss += sess.run(loss, feed_dict=feed_dict)

        avg_loss /= train_num

        # 信息显示
        if (epoch+1)%display_step == 0:
            train_acc = sess.run(accuracy, feed_dict=feed_dict)
            print("Batch: %d  loss: %.6f " % (epoch+1, avg_loss))
            print("train accuracy: %.3f" % train_acc)
            feeds = {x: x_test, y: y_test}
            test_acc = sess.run(accuracy, feed_dict=feeds)
            print("test accuracy: %.3f" % test_acc)

            if train_acc > 0.97 and test_acc > 0.97:
                feeds = {x: x_valid, y: y_valid}
                valid_acc = sess.run(accuracy, feed_dict=feeds)
                print('validation accuracy: %3f' % valid_acc)
                saver.save(sess, './lenet/mnist')
                break

        epoch += 1

# 这里是衰减learning rate代码
# global_step = tf.Variable(0, trainable=False)
# learning_rate = tf.train.exponential_decay(learning_rate=1.0,global_step=global_step,decay_steps=train_num, decay_rate=0.95,staircase=True)
