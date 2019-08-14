# -*- coding:utf-8 -*-
"""
Created on 2018/11/8 21:01

@ Author : zhl
"""

# import os
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import random
import nibabel as nib
from scipy.ndimage import affine_transform
import options3d_ACPC


width = options3d_ACPC.ImageW
height = options3d_ACPC.ImageH
depth = options3d_ACPC.ImageD # 150
channel = options3d_ACPC.CHANNELS
norm = options3d_ACPC.norm
batch_index = 0
batch_valindex = 0
filenames = []
valfilenames = []
# # user selection
num_class = options3d_ACPC.InpCol


def get_datalabel(data_dir, file_dir, acpc_dir):
    global filenames

    idx = 0
    f = open(data_dir+acpc_dir)
    line = f.readline()
    while line:
        name, ax, ay, az, px, py, pz, tx, ty, tz = line.split()
        IXI_Path = data_dir + file_dir + '/' + name
        IXI_data = nib.load(IXI_Path)
        IXI_img = IXI_data.get_data()
        filenames.append([IXI_img, [ax, ay, az, px, py, pz, tx, ty, tz, 1], IXI_Path])
        # idx+=1
        # if idx > 10 :
        #     break
        line = f.readline()

    f.close()
    # random.shuffle(filenames)  #shuffle................................................................

# get_datalabel(data_dir, file_dir, acpc_dir)

def get_testdata_MRI(data_dir, file_dir, acpc_dir, batch_size, theta, Bata):
    global batch_index, filenames#, index_test

    if len(filenames) == 0 :   #一次性读入内存
        get_datalabel(data_dir, file_dir, acpc_dir)
    max = len(filenames)

    begin = batch_index * batch_size
    end = begin + batch_size

    if end > max:
        begin = 0
        end = begin + batch_size
        batch_index = 0

    x_data = np.zeros((batch_size, height,  width,  depth, channel), np.float32)
    y_data = np.zeros((batch_size, num_class), dtype=np.int32)  # zero-filled list for 'one hot encoding'
    index = 0
    for i in range(begin, end):
        # i = (index_test * batch_size + i) % len(filenames)
        i = (begin + index) % len(filenames)
        if batch_index == 0:
            random.shuffle(filenames)
        FA_data0 = filenames[i][0]
        # rotate image
        tfAY = np.array([[np.cos(theta), 0, -np.sin(theta), 0],  # 绕Y轴旋转的矩阵
                         [0, 1.0000, 0, 0],
                         [np.sin(theta), 0, np.cos(theta), 0],
                         [0, 0, 0, 1.0000]])
        tfAZ = np.array([[np.cos(Bata), np.sin(Bata), 0, 0],  # 绕Z轴旋转的矩阵
                         [-np.sin(Bata), np.cos(Bata), 0, 0],
                         [0, 0, 1.0000, 0],
                         [0, 0, 0, 1.0000]])
        tfA1 = np.dot(tfAY, tfAZ)
        FA_data = affine_transform(FA_data0, tfA1, offset=[0, 0, 0],
                                        order=3,  # output_shape=(256, 256, 150),
                                        output=np.float32, mode='constant')
        #rotate label
        ACX = int(filenames[i][1][0])
        ACY = int(filenames[i][1][1])
        ACZ = int(filenames[i][1][2])
        PCX = int(filenames[i][1][3])
        PCY = int(filenames[i][1][4])
        PCZ = int(filenames[i][1][5])
        tfAC = np.round(np.dot([ACX, ACY, ACZ, 1], tfA1))
        tfPC = np.round(np.dot([PCX, PCY, PCZ, 1], tfA1))

        # #plot画图
        # plt.figure()
        # plt.subplot(2, 2, 1)
        # plt.imshow(FA_data0[:, :, ACZ], cmap='gray')  # X和Y坐标是反的, #坐标系变了，所以要颠倒XY坐标的顺序
        # plt.scatter(ACY, ACX, color='g', marker='v')
        # plt.subplot(2, 2, 2)
        # plt.imshow(FA_data0[:, :, PCZ], cmap='gray')
        # plt.scatter(PCY, PCX, color='g', marker='v')
        # plt.subplot(2, 2, 3)
        # plt.imshow(FA_data[:, :, int(tfAC[2])], cmap='gray')
        # plt.scatter(int(tfAC[1]), int(tfAC[0]), color='r', marker='+')
        # plt.subplot(2, 2, 4)
        # plt.imshow(FA_data[:, :, int(tfPC[2])], cmap='gray')
        # plt.scatter(int(tfPC[1]), int(tfPC[0]), color='r', marker='+')
        # plt.show()
        # print(ACX, ACY, ACZ, PCX, PCY, PCZ)

        # TensorShape([Dimension(256), Dimension(256), Dimension(150)])
        # resized_image = tf.image.resize_images(images=FA_data, size=(width, height), method=1)

        # image = sess.run(resized_image)  # (256,256,150)

        #############################################################
        #  dayta standardlized

        # p99 = np.percentile(x_data, 99)
        # x_data[x_data > p99] = p99
        # x_data = x_data / p99 * 255

        max_data = np.max(FA_data)
        FA_data = FA_data / max_data


        # x_data = np.append(x_data, np.reshape(FA_data, (height,  width,  depth, channel)))  # (image.data, dtype='float32')
        FA_data = FA_data[:, :, :, np.newaxis]
        x_data[index] = FA_data
        y_data[index][0] = tfAC[0]  # assign 1 to corresponding column (one hot encoding)
        y_data[index][1] = tfAC[1]
        y_data[index][2] = tfAC[2]
        y_data[index][3] = tfPC[0]
        y_data[index][4] = tfPC[1]
        y_data[index][5] = tfPC[2]
        index += 1

    # batch_index += batch_size  # update index for the next batch
    # x_data_ = x_data.reshape(batch_size, height,  width,  depth, channel)
    y_data = y_data
    batch_index += 1
    #print("-------------------------------")

    return x_data, y_data, filenames[:][2]

def get_data_MRI(data_dir, file_dir, acpc_dir, batch_size):
    global batch_index, filenames#, index_test

    if len(filenames) == 0 :   #一次性读入内存
        get_datalabel(data_dir, file_dir, acpc_dir)
    max = len(filenames)

    begin = batch_index * batch_size
    end = begin + batch_size

    if end > max:
        begin = 0
        end = begin + batch_size
        batch_index = 0

    x_data = np.zeros((batch_size, height,  width,  depth, channel), np.float32)
    y_data = np.zeros((batch_size, num_class), dtype=np.int32)  # zero-filled list for 'one hot encoding'
    index = 0
    for i in range(begin, end):
        # i = (index_test * batch_size + i) % len(filenames)
        i = (begin + index) % len(filenames)
        if batch_index == 0:
            random.shuffle(filenames)
        FA_data0 = filenames[i][0]
        # rotate image
        theta = np.random.randint(-5, 5, dtype=np.int32)
        # print("Y旋转角度为：", theta, "°")
        theta = theta * np.pi / 180

        Bata = np.random.randint(-10, 10, dtype=np.int32)
        # print("Z旋转角度为：", Bata, "°")
        Bata = Bata * np.pi / 180
        tfAY = np.array([[np.cos(theta), 0, -np.sin(theta), 0],  # 绕Y轴旋转的矩阵
                         [0, 1.0000, 0, 0],
                         [np.sin(theta), 0, np.cos(theta), 0],
                         [0, 0, 0, 1.0000]])

        tfAZ = np.array([[np.cos(Bata), np.sin(Bata), 0, 0],  # 绕Z轴旋转的矩阵
                         [-np.sin(Bata), np.cos(Bata), 0, 0],
                         [0, 0, 1.0000, 0],
                         [0, 0, 0, 1.0000]])

        tfA1 = np.dot(tfAY, tfAZ)
        FA_data = affine_transform(FA_data0, tfA1, offset=[0, 0, 0],
                                        order=3,  # output_shape=(256, 256, 150),
                                        output=np.float32, mode='constant')
        #rotate label
        ACX = int(filenames[i][1][0])
        ACY = int(filenames[i][1][1])
        ACZ = int(filenames[i][1][2])
        PCX = int(filenames[i][1][3])
        PCY = int(filenames[i][1][4])
        PCZ = int(filenames[i][1][5])
        TCX = int(filenames[i][1][6])
        TCY = int(filenames[i][1][7])
        TCZ = int(filenames[i][1][8])

        tfAC = np.round(np.dot([ACX, ACY, ACZ, 1], tfA1))
        tfPC = np.round(np.dot([PCX, PCY, PCZ, 1], tfA1))
        tfTC = np.round(np.dot([TCX, TCY, TCZ, 1], tfA1))

        max_data = np.max(FA_data)
        FA_data = FA_data / max_data

        # FA_data = np.transpose(FA_data, axes=(1,0, 2))
        # FA_data = np.rot90(FA_data, k=2)

        # plt.figure()
        # plt.imshow(FA_data0[100, :, :], cmap='gray')  # X和Y坐标是反的, #坐标系变了，所以要颠倒XY坐标的顺序
        # plt.show()

        # # plot画图
        # plt.figure()
        # plt.subplot(2, 2, 1)
        # plt.imshow(FA_data0[:, :, ACZ], cmap='gray')  # X和Y坐标是反的, #坐标系变了，所以要颠倒XY坐标的顺序
        # plt.scatter(ACY, ACX, color='g', marker='v')
        # plt.subplot(2, 2, 2)
        # plt.imshow(FA_data0[:, :, PCZ], cmap='gray')
        # plt.scatter(PCY, PCX, color='g', marker='v')
        # plt.subplot(2, 2, 3)
        # plt.imshow(FA_data[:, :, int(tfAC[2])], cmap='gray')
        # plt.scatter(int(tfAC[0]), int(tfAC[1]), color='r', marker='+')
        # plt.subplot(2, 2, 4)
        # plt.imshow(FA_data[:, :, int(tfPC[2])], cmap='gray')
        # plt.scatter(int(tfPC[0]), int(tfPC[1]), color='r', marker='+')
        # plt.show()

        #print(ACX, ACY, ACZ, PCX, PCY, PCZ)

        # TensorShape([Dimension(256), Dimension(256), Dimension(150)])
        # resized_image = tf.image.resize_images(images=FA_data, size=(width, height), method=1)

        # image = sess.run(resized_image)  # (256,256,150)

        #############################################################
        #  dayta standardlized

        # p99 = np.percentile(x_data, 99)
        # x_data[x_data > p99] = p99
        # x_data = x_data / p99 * 255

        # x_data = np.append(x_data, np.reshape(FA_data, (height,  width,  depth, channel)))  # (image.data, dtype='float32')

        FA_data = FA_data[:, :, :, np.newaxis]
        x_data[index] = FA_data
        y_data[index][0] = tfAC[0]  # assign 1 to corresponding column (one hot encoding)
        y_data[index][1] = tfAC[1]
        y_data[index][2] = tfAC[2]
        y_data[index][3] = tfPC[0]
        y_data[index][4] = tfPC[1]
        y_data[index][5] = tfPC[2]

        y_data[index][6] = tfTC[0]
        y_data[index][7] = tfTC[1]
        y_data[index][8] = tfTC[2]

        index += 1

    # batch_index += batch_size  # update index for the next batch
    # x_data_ = x_data.reshape(batch_size, height,  width,  depth, channel)
    y_data = y_data
    batch_index += 1
    #print("-------------------------------")

    return x_data, y_data, filenames[:][2]

# valid_dir = 'E:/deepLearning/tensorflow/graduation project/3D_ACPCLocation/ACPCChannelSE/'
# valid_data = 'data/testACPCdata150/'
# valid_label = 'data/testACPC150Label.txt'
#
#
# i = 0
# while i < 32:
#     x_, y_, _ = get_data_MRI(valid_dir, valid_data, valid_label, 2)
#     i += 1

def get_validlabel(data_dir, file_dir, acpc_dir):
    global valfilenames

    f = open(data_dir+acpc_dir)
    line = f.readline()
    while line:
        name, ax, ay, az, px, py, pz, tx, ty, tz = line.split()
        IXI_Path = data_dir + file_dir + '/' + name
        IXI_data = nib.load(IXI_Path)
        IXI_img = IXI_data.get_data()
        valfilenames.append([IXI_img, [ax, ay, az, px, py, pz, tx, ty, tz, 1], IXI_Path])
        line = f.readline()
    f.close()

def getvaliddata(data_dir, file_dir, acpc_dir, batch_size):
    global batch_valindex, valfilenames#, index_val

    if len(valfilenames) == 0 :   #一次性读入内存
        get_validlabel(data_dir, file_dir, acpc_dir)
    max = len(valfilenames)

    begin = batch_valindex * batch_size
    end = begin + batch_size
    if end > max:
        begin = 0
        end = begin + batch_size
        batch_valindex = 0
        # index_val = 0
    x_data = np.zeros((batch_size, height,  width,  depth, channel), np.float32)
    y_data = np.zeros((batch_size, num_class), dtype=np.int32)  # zero-filled list for 'one hot encoding'
    index = 0
    for i in range(begin, end):
        # i = (index_val * batch_size + i) % len(valfilenames)
        i = (begin + index) % len(valfilenames)
        FA_data0 = valfilenames[i][0]
        if i == 0:
            theta = 4
            # print("Y旋转角度为：", theta, "°")
            theta = theta * np.pi / 180
            Bata = 8
            # print("Z旋转角度为：", Bata, "°")
            Bata = Bata * np.pi / 180
        elif i == 1:
            theta = -4
            # print("Y旋转角度为：", theta, "°")
            theta = theta * np.pi / 180
            Bata = -8
            # print("Z旋转角度为：", Bata, "°")
            Bata = Bata * np.pi / 180
        else:
            theta = -3
            # print("Y旋转角度为：", theta, "°")
            theta = theta * np.pi / 180
            Bata = 5
            # print("Z旋转角度为：", Bata, "°")
            Bata = Bata * np.pi / 180

        tfAY = np.array([[np.cos(theta), 0, -np.sin(theta), 0],  # 绕Y轴旋转的矩阵
                         [0, 1.0000, 0, 0],
                         [np.sin(theta), 0, np.cos(theta), 0],
                         [0, 0, 0, 1.0000]])
        tfAZ = np.array([[np.cos(Bata), np.sin(Bata), 0, 0],  # 绕Z轴旋转的矩阵
                         [-np.sin(Bata), np.cos(Bata), 0, 0],
                         [0, 0, 1.0000, 0],
                         [0, 0, 0, 1.0000]])
        tfA1 = np.dot(tfAY, tfAZ)
        FA_data = affine_transform(FA_data0, tfA1, offset=[0, 0, 0],
                                        order=3,  # output_shape=(256, 256, 150),
                                        output=np.float32, mode='constant')
        #rotate label
        ACX = int(valfilenames[i][1][0])
        ACY = int(valfilenames[i][1][1])
        ACZ = int(valfilenames[i][1][2])
        PCX = int(valfilenames[i][1][3])
        PCY = int(valfilenames[i][1][4])
        PCZ = int(valfilenames[i][1][5])
        TCX = int(valfilenames[i][1][6])
        TCY = int(valfilenames[i][1][7])
        TCZ = int(valfilenames[i][1][8])
        tfAC = np.round(np.dot([ACX, ACY, ACZ, 1], tfA1))
        tfPC = np.round(np.dot([PCX, PCY, PCZ, 1], tfA1))
        tfTC = np.round(np.dot([TCX, TCY, TCZ, 1], tfA1))

        # #plot画图
        # plt.figure()
        # plt.subplot(2, 2, 1)
        # plt.imshow(FA_data0[:, :, ACZ], cmap='gray')  # X和Y坐标是反的, #坐标系变了，所以要颠倒XY坐标的顺序
        # plt.scatter(ACY, ACX, color='g', marker='v')
        # plt.subplot(2, 2, 2)
        # plt.imshow(FA_data0[:, :, PCZ], cmap='gray')
        # plt.scatter(PCY, PCX, color='g', marker='v')
        # plt.subplot(2, 2, 3)
        # plt.imshow(FA_data[:, :, int(tfAC[2])], cmap='gray')
        # plt.scatter(int(tfAC[1]), int(tfAC[0]), color='r', marker='+')
        # plt.subplot(2, 2, 4)
        # plt.imshow(FA_data[:, :, int(tfPC[2])], cmap='gray')
        # plt.scatter(int(tfPC[1]), int(tfPC[0]), color='r', marker='+')
        # plt.show()
        # print(ACX, ACY, ACZ, PCX, PCY, PCZ)
        # print(PCX, PCY, PCZ)
        # print(tfAC[0], tfAC[1], tfAC[2])
        # print(tfPC[0], tfPC[1], tfPC[2])

        # TensorShape([Dimension(256), Dimension(256), Dimension(150)])
        # resized_image = tf.image.resize_images(images=FA_data, size=(width, height), method=1)
        #
        # image = sess.run(resized_image)  # (256,256,150)

        #############################################################
        #  dayta standardlized

        # p99 = np.percentile(x_data, 99)
        # x_data[x_data > p99] = p99
        # x_data = x_data / p99 * 255

        max_data = np.max(FA_data)
        FA_data = FA_data / max_data

        # x_data = np.append(x_data, np.reshape(FA_data, (height,  width,  depth, channel)))  # (image.data, dtype='float32')
        FA_data = FA_data[:, :, :, np.newaxis]
        x_data[index] = FA_data
        y_data[index][0] = tfAC[0]  # assign 1 to corresponding column (one hot encoding)
        y_data[index][1] = tfAC[1]
        y_data[index][2] = tfAC[2]
        y_data[index][3] = tfPC[0]
        y_data[index][4] = tfPC[1]
        y_data[index][5] = tfPC[2]
        y_data[index][6] = tfTC[0]
        y_data[index][7] = tfTC[1]
        y_data[index][8] = tfTC[2]
        index += 1

    y_data = y_data
    batch_valindex += 1

    return x_data, y_data

valid_dir = '../'
valid_data = 'data/validACPCdata150'
valid_label = 'data/validACPC150Label.txt'

train_dir = '../'
train_data = "data/trainACPCdata150"
train_label = "data/trainACPC150Label.txt"


# x_, y_ = get_data_MRI(train_dir, train_data, train_label, 2)
# x_, y_ = get_data_MRI(valid_dir, valid_data, valid_label, 1)
# x_, y_ = getvaliddata(valid_dir, valid_data, valid_label, 1)
# print(x_.shape)
# print(y_.shape)
# i = 0
# while i < 6:
#     x_, y_ = getvaliddata(valid_dir, valid_data, valid_label, 1)
#     i += 1
# print(x_.shape)
# print(y_.shape)


