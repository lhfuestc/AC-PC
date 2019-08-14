# -*- coding:utf-8 -*-
"""
Created on 2019/4/5 14:08

@ Author : zhl
"""
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


def conv2d(x, filters, kernel_size, strides, activation=True):
    if activation:
        activation = tf.nn.relu
    else:
        activation = tf.nn.sigmoid
    conv = tf.layers.conv2d(x, filters, kernel_size, strides=strides, padding='same',
                            data_format='channels_last',
                            # kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                            # kernel_initializer=tf.random_normal_initializer,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            activation=activation,
                            use_bias=True)
    return conv


def pool2d(x):
    pool = tf.layers.max_pooling2d(x, 2, 2, padding='same')
    return pool


def channel_attention(inputs):
    size1 = inputs.get_shape()[1]
    # size2 = inputs.get_shape()[2]
    # size3 = inputs.get_shape()[3]
    # print("size: ", size)
    skip_conn = tf.identity(inputs, name='identity')

    strides = [1, size1, size1, 1]
    ksize = [1, size1, size1, 1]
    # inputs = tf.layers.average_pooling3d(skip_conn, pool_size=(size1, size1, size1), strides=(size1, size1, size1), name="avg_pool")
    inputs = tf.nn.avg_pool2d(skip_conn, ksize, strides, padding='SAME', name="avg_pool")

    in_maps = int(inputs.get_shape()[3])
    inputs = tf.layers.conv2d(inputs, filters=4, kernel_size=1, padding='same', activation=tf.nn.relu,
                              name='ca_conv_layer1')
    inputs = tf.layers.conv2d(inputs, filters=in_maps, kernel_size=1, padding='same', activation=tf.nn.sigmoid,
                              name='ca_conv_layer2')
    return tf.multiply(skip_conn, inputs)


def spacing_attention(inputs):
    skip_conn = tf.identity(inputs, name='identity')

    inputs = tf.layers.conv2d(inputs, filters=16, kernel_size=1, padding='same', activation=tf.nn.relu,
                              name='sa_conv_layer1')

    inputs = tf.layers.conv2d(inputs, filters=1, kernel_size=1, padding='same', activation=tf.nn.sigmoid,
                              name='sa_conv_layer2')

    return tf.multiply(skip_conn, inputs)


def CNN(inputs, isTraining):
    assert len(inputs.shape) == 4, "The dimension of inputs should be 4!"
    # in_maps = int(inputs.get_shape()[3])
    # dropprob = 0.5 if isTraining else 1

    conv1 = conv2d(inputs, 8, 1, 1)

    pool1 = pool2d(conv1)
    conv3 = conv2d(pool1, 16, 3, 1)

    pool2 = pool2d(conv3)

    sa = spacing_attention(pool2)

    sa = sa + pool2

    conv5 = conv2d(sa, 32, 1, 1)  # (None, 64, 64, 64, 32)

    conv6 = conv2d(conv5, 64, 3, 1)

    pool3 = pool2d(conv6)

    # 2d conv
    conv7 = conv2d(pool3, 64, 1, 1)

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
        units=1024,
        activation=tf.nn.relu,
        use_bias=True,
        # kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        # bias_initializer=tf.zeros_initializer(),
        bias_initializer=tf.truncated_normal_initializer()
    )

    # fcn2 = tf.layers.dense(
    #     inputs=fc1,
    #     units=1024,
    #     activation=tf.nn.relu,
    #     use_bias=True,
    #     # kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
    #     kernel_initializer=tf.contrib.layers.xavier_initializer(),
    #     # bias_initializer=tf.zeros_initializer(),
    #     bias_initializer=tf.truncated_normal_initializer()
    # )
    fcn3 = tf.layers.dense(
        inputs=fc1,
        units=6,
        activation=tf.nn.relu,
        use_bias=True,
        # kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        # bias_initializer=tf.zeros_initializer(),
        bias_initializer=tf.random_normal_initializer()
    )
    return fcn3


def moulti_cnn_lstm(inputs, isTraining):
    # bs = int(inputs.get_shape()[0])
    bs = 2
    assert len(inputs.shape) == 4, "The dimension of inputs should be 4!"
    num_split = 5
    feature_list = []
    input_list = tf.split(inputs, 5, 3)
    for i in range(num_split):
        with tf.variable_scope('CNN_Scope_%d' % i):
            inputs = CNN(input_list[0], isTraining)
            feature_list.append(inputs)
    X = tf.reshape(feature_list, [bs, num_split * 6, 1], name='sequence_reshape')
    out = cnn_lstm(X)
    return out


def cnn_lstm(inputs):
    batch_size = 2

    num_step = 30
    hidden_neural_size = 30
    hidden_layer_num = 1

    # build LSTM network

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_neural_size, forget_bias=0.0, state_is_tuple=True,
                                             activation=tf.nn.relu)

    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * hidden_layer_num, state_is_tuple=True)

    _initial_state = cell.zero_state(batch_size, tf.float32)

    out_put = []
    state = _initial_state
    with tf.variable_scope("LSTM_layer"):
        for time_step in range(num_step):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(inputs[:, time_step, :], state)
            out_put.append(cell_output)

    out_put = tf.transpose(out_put, [1, 2, 0])
    out_put = tf.reshape(out_put, [batch_size, hidden_neural_size * num_step])

    output = tf.layers.dense(
        inputs=out_put,
        units=6,
        activation=tf.nn.relu,
        use_bias=True,
        # kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        # bias_initializer=tf.zeros_initializer(),
        bias_initializer=tf.random_normal_initializer()
    )
    # output = tf.layers.dense(
    #     inputs=output,
    #     units=6,
    #     activation=tf.nn.relu,
    #     use_bias=True,
    #     # kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
    #     kernel_initializer=tf.contrib.layers.xavier_initializer(),
    #     # bias_initializer=tf.zeros_initializer(),
    #     bias_initializer=tf.random_normal_initializer()
    # )
    return output
