import tensorflow as tf
import matplotlib as mpl
from tensorflow.examples.tutorials.mnist import input_data


# 读取
mnist = input_data.read_data_sets('E:\deep_learning\data\mnist', one_hot=True)

# 构建神经网络(4层、1 input, 2 hidden，1 output)
n_unit_hidden_1 = 256  # 第一层hidden中的神经元数目
n_unit_hidden_2 = 128  # 第二层的hidden中的神经元数目
n_input = 784  # 输入的一个样本（图像）是28*28像素的
n_classes = 10  # 输出的类别数目

# 定义输入的占位符
x = tf.placeholder(tf.float32, shape=[None, n_input], name='x')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='y')

# 构建初始化的w和b，都是随机生成的
weights = {
    "w1": tf.Variable(tf.random_normal(shape=[n_input, n_unit_hidden_1], stddev=0.1), name='w1'),
    "w2": tf.Variable(tf.random_normal(shape=[n_unit_hidden_1, n_unit_hidden_2], stddev=0.1),name='w2'),
    "out": tf.Variable(tf.random_normal(shape=[n_unit_hidden_2, n_classes], stddev=0.1), name='out_w')
}
biases = {
    "b1": tf.Variable(tf.random_normal(shape=[n_unit_hidden_1], stddev=0.1), name='b1'),
    "b2": tf.Variable(tf.random_normal(shape=[n_unit_hidden_2], stddev=0.1), name='b2'),
    "out": tf.Variable(tf.random_normal(shape=[n_classes], stddev=0.1), name='out_b')
}


# 第一个隐层
hidden_1_x = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['w1']), biases['b1']))
# 第二个隐层
hidden_2_x  = tf.nn.sigmoid(tf.add(tf.matmul(hidden_1_x, weights['w2']), biases['b2']))
# 输出层，获取预测输出
act = tf.add(tf.matmul(hidden_2_x, weights['out']), biases['out'])

# 损失函数
# softmax_cross_entropy_with_logits: 计算softmax中的每个样本的交叉熵，logits预测值，labels实际值
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=act, labels=y))

# 使用梯度下降求解
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 得到预测的类别是那一个
# tf.argmax:对矩阵按行或列计算最大值对应的下标，和numpy中的一样
# tf.equal:是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
pred = tf.equal(tf.argmax(act, axis=1), tf.argmax(y, axis=1))
# 正确率
# tf.cast:是类型转换函数，把True转换为1，False转换为0
acc = tf.reduce_mean(tf.cast(pred, tf.float32))

# 初始化
init = tf.global_variables_initializer()

# 执行模型的训练
batch_size = 200  # 每次处理的图片数
display_step = 5  # 每5次迭代打印一次

# 开启会话
with tf.Session() as sess:
    # 进行数据初始化
    sess.run(init)

    # 模型保存
    saver = tf.train.Saver()
    epoch = 0
    while True:
        avg_cost = 0
        # 计算出总的批次
        total_batch = int(mnist.train.num_examples / batch_size)
        # 迭代更新
        for i in range(total_batch):
            # 获取x和y
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feeds = {x: batch_xs, y: batch_ys}
            # 模型训练
            sess.run(train, feed_dict=feeds)
            # 获取损失函数值
            avg_cost += sess.run(cost, feed_dict=feeds)

        # 计算平均损失
        avg_cost = avg_cost / total_batch

        # DISPLAY  显示误差率和训练集的正确率以此测试集的正确率
        if (epoch + 1) % display_step == 0:
            print("batch: %03d, loss: %.3f" % (epoch, avg_cost))
            feeds = {x: mnist.train.images, y: mnist.train.labels}
            train_acc = sess.run(acc, feed_dict=feeds)
            print("train accuracy: %.3f" % train_acc)
            feeds = {x: mnist.test.images, y: mnist.test.labels}
            test_acc = sess.run(acc, feed_dict=feeds)
            print("test accuracy: %.3f" % test_acc)

            if train_acc > 0.92 and test_acc > 0.92:
                # 当正确率达到一定值时，保存模型
                saver.save(sess, 'model/nn_model')
                break
        epoch += 1
