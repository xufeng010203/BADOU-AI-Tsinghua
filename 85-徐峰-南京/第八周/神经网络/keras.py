[1]
"""
将训练数据和检测数据加载到内存中(第一次运行需要下载数据，会比较慢):
train_images是用于训练系统的手写数字图片;
train_labels是用于标注图片的信息;
test_images是用于检测系统训练效果的图片；
test_labels是test_images图片对应的数字标签。
"""
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("train_images.shape = ", train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)

[2]

"""
用于测试的第一张图片打印出来
"""
digit = test_images[0]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()


[3]
"""
搭建一个有效识别图案的神经网络
1.layers:表示神经网络中的一个数据处理层。(dense:全连接层)
2.models.Sequential():表示把每一个数据处理层串联起来.
3.layers.Dense(…):构造一个数据处理层。
4.input_shape(28*28,):表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，
后面的“,“表示数组里面的每一个元素到底包含多少个数字都没有关系.
"""

from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss="categorical_crossentropy",
                metrics=['accuracy'])


[4]

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape(10000, 28 * 28)
test_images = test_images.astype('float32') / 255


from tensorflow.keras.utils import to_categorical

print("before", test_labels[0])

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after_change: ", test_labels[0])

[5]
"""
把数据输入到网络进行训练
train_images：用于训练的手写数字图片；
train_labels：对应的是图片的标记；
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
epochs:每次计算的循环是五次
"""

network.fit(train_images, train_labels, epochs=5, batch_size=128)

[6]

"""
测试数据输入， 检验网络学习后的图片识别效果
"""

test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print("test_loss: ", test_loss)
print("acc: ", test_acc)

[7]
'''
进行预测
'''
(train_image, train_Label), (test_image, test_label) = mnist.load_data()

digit = test_image[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

res = network.predict(digit.reshape((1, 28*28)))
print(res)