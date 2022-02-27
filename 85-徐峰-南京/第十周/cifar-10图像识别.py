import tensorflow as tf
import numpy as np
import time
import math
import Cifar10_data  #数据增强



max_step = 4000
batch_size = 100
num_example_for_eval = 10000
data_dir = "Cifar_data/cifar-10-batches-bin"


def variable_with_weight_loss(shape, stddev, w1):
    """

    :param shape:
    :param stddev:
    :param w1: 控制l2 loss的大小
    :return:
    """
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection("losses", weight_loss)
    return var

#对训练数据进行数据增强，测似文件不进行数据增强
images_train,labels_train=Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=True)
images_test,labels_test=Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=None)

#创建x，代表训练图片
x = tf.placeholder(tf.float32, shape=[batch_size, 24, 24, 3])# 原始图像的大小

#y_图片的实际标签
y_ = tf.placeholder(tf.int32, shape=[batch_size])


#创建第一个卷积层

#卷积核
kernel_1 = variable_with_weight_loss([5, 5, 3, 64], stddev=5e-2, w1=0.0)
#偏置
bias_1 = tf.Variable(tf.constant(0.0, shape=[64]))
#卷积运算
conv1 = tf.nn.conv2d(x, kernel_1, strides=[1,1,1,1], padding="SAME")
#激活
h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1, bias_1))
#池化
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")


#创建第二个卷积层
kernel_2 = variable_with_weight_loss([5, 5, 64, 64], stddev=5e-2, w1=0.0)
bias_2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.conv2d(h_pool1, kernel_2, strides=[1,1,1,1], padding="SAME")
h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2, bias_2))
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")


#全连接层
reshape = tf.reshape(h_pool1, shape=[batch_size, -1])#拉成一维
dim = reshape.get_shape()[1].value

#建立第一个全链接层, 384维向量
fc_weight1 = variable_with_weight_loss([dim, 384], stddev=0.04, w1=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_1 = tf.nn.relu(tf.matmul(reshape, fc_weight1) + fc_bias1)# [100, 384]

#建立第二个全连接层 拉成192
fc_weight2 = variable_with_weight_loss([384, 192],stddev=0.04, w1=0.04)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
fc_2 = tf.nn.relu(tf.matmul(fc_1, fc_weight2) + fc_bias2) #[100, 192]

#建立第三个全连接层 拉成10
fc_weight3 = variable_with_weight_loss([192, 10], stddev=1/192.0, w1=0.0)
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
fc_3 = tf.add(tf.matmul(fc_2, fc_weight3),fc_bias3) #【100， 10】

#计算损失
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc_3, labels=(tf.cast(y_, tf.int64)))

weight_with_l2_loss = tf.add_n(tf.get_collection("losses"))
loss = tf.reduce_mean(cross_entropy) + weight_with_l2_loss

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

top_k = tf.nn.in_top_k(fc_3, y_, 1)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    tf.train.start_queue_runners()

    for step in range(max_step):
        start_time = time.time()
        image_batch, label_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, loss], feed_dict={x:image_batch, y_:label_batch})
        duration = time.time() - start_time

        if step % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)"%(step,loss_value,examples_per_sec,sec_per_batch))


    #计算最终的正确率
    num_batch=int(math.ceil(num_example_for_eval/batch_size))  #math.ceil()函数用于求整
    true_count=0
    total_sample_count=num_batch * batch_size

    #在一个for循环里面统计所有预测正确的样例个数
    for j in range(num_batch):
        image_batch,label_batch=sess.run([images_test,labels_test])
        predictions=sess.run([top_k],feed_dict={x:image_batch,y_:label_batch})
        true_count += np.sum(predictions)

    #打印正确率信息
    print("accuracy = %.3f%%"%((true_count/total_sample_count) * 100))













