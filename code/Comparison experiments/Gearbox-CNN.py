import numpy as np

# import scipy.io as sio
# import math
# import time
# import matplotlib.pyplot as plt
# import scipy.signal as signal
# import scipy
# from matplotlib import cm

dataset = []
k = 4


# 加载文件
def loadDatadet(infile, k):
    f = open(infile, 'r')
    # 读取文件数据，每行是一个字符串，返回一个字符串列表
    sourceInLine = f.readlines()
    dataset = []
    # 遍历每行数据
    for line in sourceInLine:
        # 清除每行后面的换行符
        temp1 = line.strip('\n')
        # 按照制表符将数据分割
        temp2 = temp1.split('\t')
        dataset.append(temp2)
    # 返回的数据是一个二维列表
    return dataset


# 将数据归一化为0-1
def normalization(x):
    y = np.array(x)
    Max = max(y)
    Min = min(y)
    for i in range(len(x)):
        y[i] = (y[i] - Min) / (Max - Min)
    return y.tolist()
    # return x


#
p = 400
p1 = p * 10
# ----------------------crack 0-------------------
mj = 0
temp00 = []
infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-00\1500-0.txt'
# 得到原始数据的二维列表
temp1 = loadDatadet(infile, k)
# 第一行为空字符串
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    # 得到第k列数据，转换为浮点型列表
    temp00.append(float(temp1[i][k]))
temp00 = np.array(temp00)
temp00 = normalization(temp00)
for j in range(0, p1, p):
    # 将数据划分，每行p个
    dataset.append(temp00[0 + j:p + j])

temp01 = []
infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-00\1500-2.txt'
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp01.append(float(temp1[i][k]))
temp01 = np.array(temp01)
temp01 = normalization(temp01)
for j in range(0, p1, p):
    dataset.append(temp01[0 + j:p + j])

infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-00\1500-4.txt'
temp02 = []
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp02.append(float(temp1[i][k]))
temp02 = np.array(temp02)
temp02 = normalization(temp02)
for j in range(0, p1, p):
    dataset.append(temp02[0 + j:p + j])

infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-00\1500-6.txt'
temp03 = []
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp03.append(float(temp1[i][k]))
temp03 = np.array(temp03)
temp03 = normalization(temp03)
for j in range(0, p1, p):
    dataset.append(temp03[0 + j:p + j])

infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-00\1500-8.txt'
temp04 = []
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp04.append(float(temp1[i][k]))
temp04 = np.array(temp04)
temp04 = normalization(temp04)
for j in range(0, p1, p):
    dataset.append(temp04[0 + j:p + j])

infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-00\1500-10.txt'
temp05 = []
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp05.append(float(temp1[i][k]))
temp05 = np.array(temp05)
temp05 = normalization(temp05)
for j in range(0, p1, p):
    dataset.append(temp05[0 + j:p + j])

# ----------------------crack 5-------------------
mj = 1
temp10 = []
infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-01\1500-0.txt'
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp10.append(float(temp1[i][k]))
temp10 = np.array(temp10)
temp10 = normalization(temp10)
for j in range(0, p1, p):
    dataset.append(temp10[0 + j:p + j])

temp11 = []
infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-01\1500-2.txt'
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp11.append(float(temp1[i][k]))
temp11 = np.array(temp11)
temp11 = normalization(temp11)
for j in range(0, p1, p):
    dataset.append(temp11[0 + j:p + j])

infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-01\1500-4.txt'
temp12 = []
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp12.append(float(temp1[i][k]))
temp12 = np.array(temp12)
temp12 = normalization(temp12)
for j in range(0, p1, p):
    dataset.append(temp12[0 + j:p + j])

infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-01\1500-6.txt'
temp13 = []
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp13.append(float(temp1[i][k]))
temp13 = np.array(temp13)
temp13 = normalization(temp13)
for j in range(0, p1, p):
    dataset.append(temp13[0 + j:p + j])

infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-01\1500-8.txt'
temp14 = []
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp14.append(float(temp1[i][k]))
temp14 = np.array(temp14)
temp14 = normalization(temp14)
for j in range(0, p1, p):
    dataset.append(temp14[0 + j:p + j])

infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-01\1500-10.txt'
temp15 = []
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp15.append(float(temp1[i][k]))
temp15 = np.array(temp15)
temp15 = normalization(temp15)
for j in range(0, p1, p):
    dataset.append(temp15[0 + j:p + j])

# #----------------------crack 10-------------------
mj = 2
temp20 = []
infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-02\1500-0.txt'
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp20.append(float(temp1[i][k]))
temp20 = np.array(temp20)
temp20 = normalization(temp20)
for j in range(0, p1, p):
    dataset.append(temp20[0 + j:p + j])

temp21 = []
infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-02\1500-2.txt'
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp21.append(float(temp1[i][k]))
temp21 = np.array(temp21)
temp21 = normalization(temp21)
for j in range(0, p1, p):
    dataset.append(temp21[0 + j:p + j])

infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-02\1500-4.txt'
temp22 = []
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp22.append(float(temp1[i][k]))
temp22 = np.array(temp22)
temp22 = normalization(temp22)
for j in range(0, p1, p):
    dataset.append(temp22[0 + j:p + j])

infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-02\1500-6.txt'
temp23 = []
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp23.append(float(temp1[i][k]))
temp23 = np.array(temp23)
temp23 = normalization(temp23)
for j in range(0, p1, p):
    dataset.append(temp23[0 + j:p + j])

infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-02\1500-8.txt'
temp24 = []
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp24.append(float(temp1[i][k]))
temp24 = np.array(temp24)
temp24 = normalization(temp24)
for j in range(0, p1, p):
    dataset.append(temp24[0 + j:p + j])

infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-02\1500-10.txt'
temp25 = []
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp25.append(float(temp1[i][k]))
temp25 = np.array(temp25)
temp25 = normalization(temp25)
for j in range(0, p1, p):
    dataset.append(temp25[0 + j:p + j])

# #----------------------crack 15-------------------
mj = 3
temp30 = []
infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-03\1500-0.txt'
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp30.append(float(temp1[i][k]))
temp30 = np.array(temp30)
temp30 = normalization(temp30)
for j in range(0, p1, p):
    dataset.append(temp30[0 + j:p + j])

temp31 = []
infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-03\1500-2.txt'
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp31.append(float(temp1[i][k]))
temp31 = np.array(temp31)
temp31 = normalization(temp31)
for j in range(0, p1, p):
    dataset.append(temp31[0 + j:p + j])

infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-03\1500-4.txt'
temp32 = []
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp32.append(float(temp1[i][k]))
temp32 = np.array(temp32)
temp32 = normalization(temp32)
for j in range(0, p1, p):
    dataset.append(temp32[0 + j:p + j])

infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-03\1500-6.txt'
temp33 = []
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp33.append(float(temp1[i][k]))
temp33 = np.array(temp33)
temp33 = normalization(temp33)
for j in range(0, p1, p):
    dataset.append(temp33[0 + j:p + j])

infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-03\1500-8.txt'
temp34 = []
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp34.append(float(temp1[i][k]))
temp34 = np.array(temp34)
temp34 = normalization(temp34)
for j in range(0, p1, p):
    dataset.append(temp34[0 + j:p + j])

infile = r'E:\pythonCode\SuperGraph\data\Gearbox\LW-03\1500-10.txt'
temp35 = []
temp1 = loadDatadet(infile, k)
temp1 = temp1[1:p1 + 1]
for i in range(0, len(temp1)):
    temp35.append(float(temp1[i][k]))
temp35 = np.array(temp35)
temp35 = normalization(temp35)
for j in range(0, p1, p):
    dataset.append(temp35[0 + j:p + j])

s = []
freqs = []
# 把数据集赋值给s
for i in range(len(dataset)):
    y2 = dataset[i]
    s.append(y2)

y = []  # label

# 数据集240行，给y赋值60个0、1、2、3
for i in range(60):
    y.append(0)

for i in range(60):
    y.append(1)

for i in range(60):
    y.append(2)

for i in range(60):
    y.append(3)

# 把数据集赋值给image
image = s
# 数据集转换为数组 240行400列
im = np.array(image)
# 调整数据集的形状
im.shape = 240, 20, 20, 1
image = im

label = y

import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, LeakyReLU, Dropout, GlobalMaxPooling2D
from keras.optimizers import Adam

# import matplotlib.pyplot as plt
# import os
# import time
# import pandas as pd
# from PIL import Image
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from skimage import util

data = image
from sklearn.model_selection import train_test_split

# 将数据和标签随机划分为训练集和测试集，测试数据占比40%
# stratify分层抽样，保证训练和测试各样本的比例相同
X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.4, random_state=42,
                                                    stratify=label)  # divide train,test,validation


def build_CNN():  # CNN model
    model = Sequential()

    # 添加卷积层
    model.add(Convolution2D(
        # 输入形状
        batch_input_shape=(None, 20, 20, 1),
        # 卷积核的数目为16个
        filters=16,
        # 卷积核大小
        kernel_size=5,
        # 步长
        strides=1,
        # 边界进行零填充，输出与输入形状相同
        padding='same',
    ))

    # 添加ReLU变体激活函数，负半部分斜率设置为0.3
    model.add(LeakyReLU(alpha=0.3))

    # 添加池化层
    model.add(MaxPooling2D(
        # 池化核大小
        pool_size=2,
        strides=1,
        padding='same',
    ))

    # 总体期望不变，随机丢弃40%神经元
    model.add(Dropout(0.4))

    # 卷积核数目8，核函数大小8，步长2
    model.add(Convolution2D(8, 8, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(2, 2, 'same'))
    model.add(Dropout(0.4))

    model.add(Convolution2D(8, 8, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(2, 2, 'same'))
    model.add(Dropout(0.4))

    # 将多维展开为1维
    model.add(Flatten())
    # 添加全连接层，输出200个元素
    model.add(Dense(200))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(4))
    # 使用softmax函数将输出转化为0到1之间，总和为1
    model.add(Activation('softmax'))
    # 设置学习率为0.001，使用Adam优化器
    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',  # 定义损失函数为交叉熵损失函数
                  metrics=['accuracy'])  # 评估模型为准确率
    return model


# 迭代100次
epochs = 100
batch_size = 16
test_acc = []
x_train = np.array(X_train).reshape(-1, 20, 20, 1).astype('float32')
x_test = np.array(X_test).reshape(-1, 20, 20, 1).astype('float32')
# 根据标签进行独热编码
y_train = np_utils.to_categorical(Y_train, num_classes=4)
y_test = np_utils.to_categorical(Y_test, num_classes=4)
# 将训练数据集和测试数据集进行特征标椎化
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean
model = build_CNN()
# 输出模型信息
model.summary()
acc = []

for i in range(10):  # diagnosis
    # validation_split使用一半的训练数据作为验证数据
    # epochs训练轮数 batch_size每次训练使用的样本数量 verbose每轮迭代后输出一行
    history = model.fit(x_train, y_train, validation_split=0.5, epochs=epochs, batch_size=batch_size, verbose=1)
    # 评估损失和准确度
    loss, accuracy = model.evaluate(x_test, y_test)
    acc.append([accuracy])
    print(accuracy * 100)
