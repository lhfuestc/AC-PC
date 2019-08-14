# -*- coding:utf-8 -*-
"""
Created on 2019/3/28 14:04

@ Author : zhl
"""
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
from datetime import datetime
import time
import math

def conv3d(x, filters, kernel_size, strides):
    """
    :param x:  input
    :param filters: filter的个数
    :param kernel_size: 整数或3维的tuple，filter的大小
    :return:
    """
    conv = tf.layers.conv3d(x, filters, kernel_size, strides=strides, padding='same',
                            data_format = 'channels_last',
                            #kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                           # kernel_initializer=tf.random_normal_initializer,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            activation= tf.nn.relu,
                            #activation = tf.nn.relu,
                            use_bias=True)

    return conv


def pool3d(x):
    pool = tf.layers.max_pooling3d(x, 2, 2, padding='same')
    return pool



def conv2d(x, filters, kernel_size, strides, activation = True):
    if activation:
        activation = tf.nn.relu
    else:
        activation = tf.nn.sigmoid
    conv = tf.layers.conv2d(x, filters, kernel_size, strides=strides, padding='same',
                            data_format='channels_last',
                            #kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                            # kernel_initializer=tf.random_normal_initializer,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            activation=activation,
                            use_bias=True)
    return conv


def pool2d(x):
    pool = tf.layers.max_pooling2d(x, 2, 2, padding='same')
    return pool


def channel_attention(inputs):
    """
    空间注意力机制
    :param inputs:
    :return:
    """
    size1 = inputs.get_shape()[1]
    #size2 = inputs.get_shape()[2]
    #size3 = inputs.get_shape()[3]
    # print("size: ", size)
    skip_conn = tf.identity(inputs, name='identity')

    strides = [1, size1, size1, size1, 1]
    ksize = [1, size1, size1, size1, 1]
    #inputs = tf.layers.average_pooling3d(skip_conn, pool_size=(size1, size1, size1), strides=(size1, size1, size1), name="avg_pool")
    inputs = tf.nn.avg_pool3d(skip_conn, ksize, strides, padding='SAME', name="avg_pool")

    in_maps = int(inputs.get_shape()[4])
    inputs = tf.layers.conv3d(inputs, filters=4, kernel_size=1, padding='same', activation=tf.nn.relu,
                              name='ca_conv_layer1')
    inputs = tf.layers.conv3d(inputs, filters=in_maps, kernel_size=1, padding='same', activation=tf.nn.sigmoid,
                              name='ca_conv_layer2')
    return tf.multiply(skip_conn, inputs)

def spacing_attention(inputs):
    """
    空间注意力机制
    """
    skip_conn = tf.identity(inputs, name='identity')

    inputs = tf.layers.conv3d(inputs, filters=16, kernel_size=1, padding='same', activation=tf.nn.relu,
                              name='sa_conv_layer1')

    inputs = tf.layers.conv3d(inputs, filters=1, kernel_size=1, padding='same', activation=tf.nn.sigmoid,
                              name='sa_conv_layer2')

    return tf.multiply(skip_conn, inputs)


def Model_3D(inputs, isTraining):
    assert len(inputs.shape) == 5, "The dimension of inputs should be 5! "

    # dropprob = 0.5 if isTraining else 1

    conv1 = conv3d(inputs, 8, 1, 1)

    pool1 = pool3d(conv1)
    conv3 = conv3d(pool1, 16, 3, 1)

    pool2 = pool3d(conv3)

    sa = spacing_attention(pool2)

    sa = sa + pool2

    conv5 = conv3d(sa, 32, 1, 1)  # (None, 64, 64, 64, 32)

    conv6 = conv3d(conv5, 64, 3, 1)

    pool3 = pool3d(conv6)

    # 通道相加
    ca = channel_attention(pool3)
    ca = ca + pool3

    ca2d = tf.reduce_sum(ca, 4)

    # 2d conv
    conv7 = conv2d(ca2d, 64, 1, 1)

    conv8 = conv2d(conv7, 128, 3, 1)

    pool4 = pool2d(conv8)

    conv9 = conv2d(pool4, 256, 3, 1, activation=False)
    # conv9 = conv2d(pool4, 256, 3, 1)

    pool5 = pool2d(conv9)

    #    flatten = tf.layers.flatten(conv5)
    size_input1d = int(pool5.get_shape()[1]) * int(pool5.get_shape()[2]) * int(pool5.get_shape()[3])
    flatten = tf.reshape(pool5, [-1, size_input1d], name="reshape")

    fc1 = tf.layers.dense(
        inputs=flatten,
        units=4096,
        activation=tf.nn.relu,
        use_bias=True,
        # kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        # bias_initializer=tf.zeros_initializer(),
        bias_initializer=tf.truncated_normal_initializer()
    )

    fcn2 = tf.layers.dense(
        inputs=fc1,
        units=1024,
        activation=tf.nn.relu,
        use_bias=True,
        # kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        # bias_initializer=tf.zeros_initializer(),
        bias_initializer=tf.truncated_normal_initializer()
    )
    fcn3 = tf.layers.dense(
        inputs=fcn2,
        units=9,
        activation=tf.nn.relu,
        use_bias=True,
        # kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        # bias_initializer=tf.zeros_initializer(),
        bias_initializer=tf.random_normal_initializer()
    )
    return fcn3


