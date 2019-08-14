# -*- coding:utf-8 -*-
"""
Created on 2018/11/4 21:25

@ Author : zhl
"""

import tensorflow as tf
import options3d_ACPC
import numpy as np

TOTAL_LOSS_COLLECTION = options3d_ACPC.TOTAL_LOSS_COLLECTION
BATCH_SIZE = options3d_ACPC.params.batch_size
pi = options3d_ACPC.PI
lamd = options3d_ACPC.lamd

def cal_theta(real_AP, pred_AP):
    """
    :param real_AP:
    :param pred_AP:
    :return: cal theta of the pred_ACPCline and the true_ACPCline
    """
    real_AP = tf.cast(real_AP, dtype=tf.float32)
    pred_AP = tf.cast(pred_AP, dtype=tf.float32)
    real_AC = tf.slice(real_AP, [0, 0], [BATCH_SIZE, 3])
    real_PC = tf.slice(real_AP, [0, 3], [BATCH_SIZE, 3])
    ACPCTrueline = tf.subtract(real_AC, real_PC)

    pred_AC = tf.slice(pred_AP, [0, 0], [BATCH_SIZE, 3])
    pred_PC = tf.slice(pred_AP, [0, 3], [BATCH_SIZE, 3])
    ACPCPredline = tf.subtract(pred_AC, pred_PC)

    True_norm = tf.sqrt(tf.reduce_sum(tf.square(ACPCTrueline), axis=1))
    Pred_norm = tf.sqrt(tf.reduce_sum(tf.square(ACPCPredline), axis=1))
    multisum = tf.reduce_sum(tf.multiply(ACPCTrueline, ACPCPredline), axis=1)
    cosin = multisum / (True_norm * Pred_norm + 0.0001)
    # cosin = tf.transpose(cosin, name="cosin")
    theta = tf.acos(cosin, name="theta")
    # theta = np.reshape(theta, [BATCH_SIZE, 1])
    theta = tf.reduce_mean(theta)

    # if cosin < 0:
    #     theta = tf.subtract(tf.constant(pi, tf.float32), theta, name="theta")
    return theta


def loss_l2(real_AP, pred_AP):
    """Calculates the L2 loss from the real ACPC point and the predicted ACPC point.
            total_loss = ||real_AP - pred_AP||_L2 + η·WD
                       = ||real_AP - pred_AP||_2^2 + η·WD
       where 'WD' indicates weight decay.
    Args:
        real_AP: The ground-truth AP point with shape of [batch_size, height, width, channels]
        pred_AP: The predicated AP point by model with the same shape as real_AP

    Returns:
        total_loss: MSE between real AP point and predicted AP point and weight decay(optional).
    """
    with tf.name_scope('l2_loss'):
        # theta = cal_theta(real_AP, pred_AP)
        # ???
        # mse_loss = tf.losses.mean_squared_error(real_AP, pred_AP) + lamd * theta
        mse_loss = tf.losses.mean_squared_error(real_AP, pred_AP)
        #tf.add_to_collection(TOTAL_LOSS_COLLECTION, mse_loss)

        #  #Attach a scalar summary to mse_loss
        #tf.summary.scalar(mse_loss.op.name, mse_loss)
    #return tf.add_n(tf.get_collection(TOTAL_LOSS_COLLECTION), name='total_loss_l2')
    return mse_loss


def loss_l1(real_AP, pred_AP):
    """Calculates the L1 loss from the real AP point and the predicted AP point.
              total_loss = ||real_AP - pred_AP||_L1 + η·WD
       where 'WD' indicates weight decay.
    Args:
        real_AP: The ground-truth AP point with shape of [batch_size, height, width, channels]
        pred_AP: The predicated AP point by model with the same shape as real_AP
    Returns:
        total loss: L1 loss between real AP point and predicted AP point.
    """
    with tf.name_scope('l1_loss'):
        abs_loss = tf.reduce_mean(tf.abs(real_AP - pred_AP))
        tf.add_to_collection(TOTAL_LOSS_COLLECTION, abs_loss)
        tf.summary.scalar(abs_loss.op.name, abs_loss)

    return tf.add_n(tf.get_collection(TOTAL_LOSS_COLLECTION))


def loss_lp(real_AP, pred_AP, p):
    """Calculates the Lp loss from the real AP point and the predicted AP point.
              total_loss = ||real_AP - pred_AP||_Lp + η·WD
       where 'WD' indicates weight decay. p should be smaller than 1.0
       --> try: (2/3)*[x^(3/2)]
    Args:
        real_AP: The ground-truth AP point with shape of [batch_size, height, width, channels]
        pred_AP: The predicated AP point by model with the same shape as real_AP
        p: the factor of the loss
    Returns:
        total loss: Lp loss between real Ap point and predicted AP point.
    """

    alpha = 1e-2
    with tf.name_scope('lp_loss'):
        lp_loss = tf.reduce_mean(tf.pow(tf.abs(real_AP - pred_AP) + alpha, p))
        tf.add_to_collection(TOTAL_LOSS_COLLECTION, lp_loss)
        tf.summary.scalar(lp_loss.op.name, lp_loss)

    return tf.add_n(tf.get_collection(TOTAL_LOSS_COLLECTION))


def loss_cross(real_AP, pred_AP):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred_AP, labels=real_AP,
                                                            name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    return cross_entropy_mean