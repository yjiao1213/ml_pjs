import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf


# 读取mnist数据
dir = "E:\deep_learning\data\mnist"
mnist = input_data.read_data_sets(dir, one_hot=True)
print(mnist)


# 取train，test和validation数据集
train_x = mnist.train.images # (55000, 784)
train_y = mnist.train.labels # (55000, 10)

test_x = mnist.test.images  # (10000, 784)
test_y = mnist.test.labels  # (10000, 10)

val_x = mnist.validation.images  # (5000, 784)
val_y = mnist.validation.labels  # (5000, 10)


# 随机显示5张照片
nsample = 5
idx = np.random.randint(0, mnist.train.images.shape[0], nsample)

for i in idx:
    # reshape：格式变化, 变为28*28的矩阵图像
    curr_img = np.reshape(train_x[i, :], (28, 28))
    # 获取最大值,就是当前图像的label（10个数字中，只有一个为1，其它均为0，所以最大值极为数字对应的实际值）
    curr_label = np.argmax(train_y[i, :])
    # 矩阵图
    plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
    plt.title("NO" + str(i) + "graph, label：" + str(curr_label))
    plt.show()


# 使用tf构建softmax模型, 为x，y添加占位符，在训练时才fetch进入数据，为行向量填None
x = tf.placeholder(dtype=tf.float32, shape= [None, 784], name="x")
y = tf.placeholder(tf.float32, [None, 10], "y")

# 建立weight和bias
w = tf.Variable(initial_value=tf.zeros([784, 10]), name="w")
b = tf.Variable(initial_value=tf.zeros([10]), name="b")

# 计算预测值
y_hat = tf.add(tf.matmul(x, w), b)

#并使用softmax作为激活函数
actv = tf.nn.softmax(y_hat)

# 计算loss
# reduce_sum 矩阵和
# reduce_mean,求均值，当不给定任何axis参数的时候，表示求解全部所有数据的均值
loss = - tf.reduce_mean(tf.reduce_sum(y * tf.log(actv), axis=1))
# 使用梯度下降求解,还有多种优化器，在train中
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# 计算正确率
# tf.argmax:对矩阵按行或列计算最大值
# tf.equal:是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False
equl = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))
# 正确率
accr = tf.reduce_mean(tf.cast(equl, "float"))
# 设置初始化
init = tf.global_variables_initializer()

# 总训练次数
training_epochs = 50
# batch大小，也就是一次训练时分成多少组
batch_size = 100
# 训练时，每迭代5次，显示一次信息
display_step = 5
# 创建会话
with tf.Session() as sess:
    # 训练前进行初始化
    sess.run(init)

    for epoch in range(training_epochs):

        # 统计每个batch的loss
        avg_loss = 0
        # 55000/100
        num_batch = int(mnist.train.num_examples/batch_size)

        for i in range(num_batch):
            # 获取数据集 next_batch获取下一批的数据
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 模型训练
            feeds = {x: batch_xs, y: batch_ys}   # 每次给100个数据进行训练
            sess.run(train, feed_dict=feeds)     # 训练
            # 每次batch的loss加在一起求平均才是最终的loss
            avg_loss += sess.run(loss, feed_dict=feeds) / num_batch

        # 每迭代五次，输出训练的准确率和测试集的准确率
        if epoch % display_step == 0:
            feeds_train = {x: batch_xs, y: batch_ys}
            feeds_test = {x: mnist.test.images, y: mnist.test.labels}
            train_acc = sess.run(accr, feed_dict=feeds_train)
            test_acc = sess.run(accr, feed_dict=feeds_test)
            print("Epoch %d/%d, loss: %3f, train accuracy: %3f, test accuracy: %3f" %
                  (epoch, training_epochs, avg_loss, train_acc, test_acc))

print("Done")
# Epoch 0/50, loss: 1.177024, train accuracy: 0.860000, test accuracy: 0.854700
# Epoch 5/50, loss: 0.440981, train accuracy: 0.910000, test accuracy: 0.895200
# Epoch 10/50, loss: 0.383375, train accuracy: 0.840000, test accuracy: 0.905100
# Epoch 15/50, loss: 0.357309, train accuracy: 0.880000, test accuracy: 0.909100
# Epoch 20/50, loss: 0.341437, train accuracy: 0.930000, test accuracy: 0.912100
# Epoch 25/50, loss: 0.330506, train accuracy: 0.940000, test accuracy: 0.914300
# Epoch 30/50, loss: 0.322383, train accuracy: 0.910000, test accuracy: 0.915300
# Epoch 35/50, loss: 0.315964, train accuracy: 0.950000, test accuracy: 0.916300
# Epoch 40/50, loss: 0.310721, train accuracy: 0.930000, test accuracy: 0.916800
# Epoch 45/50, loss: 0.306360, train accuracy: 0.880000, test accuracy: 0.918700