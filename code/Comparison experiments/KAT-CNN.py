# import psutil
# import os
#
# import numpy as np
import scipy.io as sio
# import math
# import time
# from scipy.fftpack import fft, fftshift, ifft
# from scipy.fftpack import fftfreq
# import matplotlib.pyplot as plt
import numpy as np


# import scipy.signal as signal
# import scipy
# from matplotlib import cm


def read_mat(path, key):
    data = sio.loadmat(path)
    return (data[key])


def splitlist(list1):
    alist = []
    a = 0
    # 把300行一列的标签数据转化为1行300列列表
    for sublist in list1:
        try:
            for i in sublist:
                alist.append(i)
        except TypeError:
            alist.append(sublist)
    for i in alist:
        # 如果i是列表，这里i是单个数据，并非列表
        if type(i) == type([]):
            a = +1
            break
    if a == 1:
        return splitlist(alist)
    if a == 0:
        return alist


def arr_size(arr, size):
    s = []
    for i in range(0, int(len(arr)) + 1, size):
        c = arr[i:i + size]
        s.append(c)
    return s


# read data
# ---------------------------0 load--------------------------------
path = 'E:\pythonCode\SuperGraph\data\KAT\KATData0.mat'
key = 'data'
data01 = read_mat(path, key)
# 读取数据300,2560
data01 = data01.tolist()
# 每个标签下获取10行数据
datas01 = data01[0:10] + data01[100:110] + data01[200:210]
key = 'label'
y01 = read_mat(path, key)
y01 = splitlist(y01)
# 获取对应数据的标签
ys01 = y01[0:10] + y01[100:110] + y01[200:210]

# ---------------------------1 load--------------------------------
path = 'E:\pythonCode\SuperGraph\data\KAT\KATData1.mat'
key = 'data'
data02 = read_mat(path, key)
data02 = data02.tolist()
datas02 = data02[0:10] + data02[100:110] + data02[200:210]
key = 'label'
y02 = read_mat(path, key)
y02 = splitlist(y02)
ys02 = y02[0:10] + y02[100:110] + y02[200:210]

# ---------------------------2 load--------------------------------
path = 'E:\pythonCode\SuperGraph\data\KAT\KATData2.mat'
key = 'data'
data03 = read_mat(path, key)
data03 = data03.tolist()
datas03 = data03[0:10] + data03[100:110] + data03[200:210]
key = 'label'
y03 = read_mat(path, key)
y03 = splitlist(y03)
ys03 = y03[0:10] + y03[100:110] + y03[200:210]

# ---------------------------3 load--------------------------------
path = 'E:\pythonCode\SuperGraph\data\KAT\KATData3.mat'
key = 'data'
data04 = read_mat(path, key)
data04 = data04.tolist()
datas04 = data04[0:10] + data04[100:110] + data04[200:210]
key = 'label'
y04 = read_mat(path, key)
y04 = splitlist(y04)
ys04 = y04[0:10] + y04[100:110] + y04[200:210]

# 将各个数据集获取到的数据添加到一起，共120行，2560列
image = []
for i in datas01:
    image.append(i)
for i in datas02:
    image.append(i)
for i in datas03:
    image.append(i)
for i in datas04:
    image.append(i)
dataa = []
# 获取数据的前2500列
for i in image:
    dataa.append(i[0:2500])
image = dataa
im = np.array(image)
im.shape = 120, 50, 50, 1
image = im
y = []  # label
for x in ys01:
    y.append(x)
for x in ys02:
    y.append(x)
for x in ys03:
    y.append(x)
for x in ys04:
    y.append(x)

# 标签由1,2,3改为0,1,2
y = [x - 1 for x in y]
label = y

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

X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.6, random_state=42,
                                                    stratify=label)  # divide train,test,validation


def build_CNN():  # CNN model
    model = Sequential()

    model.add(Convolution2D(
        batch_input_shape=(None, 50, 50, 1),
        filters=16,
        kernel_size=5,
        strides=1,
        padding='same',
    ))

    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(
        pool_size=2,
        strides=1,
        padding='same',
    ))
    model.add(Dropout(0.4))

    model.add(Convolution2D(8, 8, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(2, 2, 'same'))
    model.add(Dropout(0.4))

    model.add(Convolution2D(8, 8, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(2, 2, 'same'))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(16))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


epochs = 100
batch_size = 16
test_acc = []
x_train = np.array(X_train).reshape(-1, 50, 50, 1).astype('float32')
x_test = np.array(X_test).reshape(-1, 50, 50, 1).astype('float32')
# 独热编码
y_train = np_utils.to_categorical(Y_train, num_classes=3)
y_test = np_utils.to_categorical(Y_test, num_classes=3)

# 特征标椎化
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean
model = build_CNN()
model.summary()
acc = []

for i in range(10):
    history = model.fit(x_train, y_train, validation_split=0.5, epochs=epochs, batch_size=batch_size, verbose=1)
    loss, accuracy = model.evaluate(x_test, y_test)
    acc.append([accuracy])
    print(accuracy * 100)
