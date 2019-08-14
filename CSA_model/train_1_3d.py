# -*- coding:utf-8 -*-
"""
Created on 2018/11/8 20:59

@ Author : zhl
"""

import input_3DImage
import tensorflow as tf
import model_se_3Dto2D as md3
from datetime import datetime
import numpy as np
import optimize3d_ACPC
import options3d_ACPC
import termcolor
import warnings
import loss3d_ACPC
import time
import os
import math

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_dir = options3d_ACPC.DEFAULT_DATA_DIR
train_image_dir = options3d_ACPC.params.train_image_dir
train_lab_dir = options3d_ACPC.params.train_label_dir
valid_dir = options3d_ACPC.valid_dir
valid_data = options3d_ACPC.valid_data
valid_label = options3d_ACPC.valid_label


num_examples = 12970
EPOCH = 100000

img_H = options3d_ACPC.ImageH
img_W = options3d_ACPC.ImageW
img_D = options3d_ACPC.ImageD
nChann = options3d_ACPC.CHANNELS
nLabel = options3d_ACPC.InpCol
BATCH_SIZE = options3d_ACPC.params.batch_size
norm =options3d_ACPC.norm
pi = options3d_ACPC.PI

train_log_dir = options3d_ACPC.params.train_log_dir
learning_rate = options3d_ACPC.params.learning_rate
Batch_size = options3d_ACPC.params.batch_size
max_steps = options3d_ACPC.params.max_steps

train_from_exist = options3d_ACPC.params.train_from_exist
exist_model_dir = options3d_ACPC.params.exist_model_dir


# ACPCTC相对误差。
def TRE(lab_batch, pred_batch):
    num = lab_batch.shape[0]
    batch_tre = np.abs(pred_batch - lab_batch) / (lab_batch + 0.0001)
    TAE_AC_X = 0.0
    TAE_AC_Y = 0.0
    TAE_AC_Z = 0.0
    TAE_PC_X = 0.0
    TAE_PC_Y = 0.0
    TAE_PC_Z = 0.0
    TAE_TC_X = 0.0
    TAE_TC_Y = 0.0
    TAE_TC_Z = 0.0
    for i in range(num):
        TAE_AC_X += batch_tre[i, 0]
        TAE_AC_Y += batch_tre[i, 1]
        TAE_AC_Z += batch_tre[i, 2]
        TAE_PC_X += batch_tre[i, 3]
        TAE_PC_Y += batch_tre[i, 4]
        TAE_PC_Z += batch_tre[i, 5]
        TAE_TC_X += batch_tre[i, 6]
        TAE_TC_Y += batch_tre[i, 7]
        TAE_TC_Z += batch_tre[i, 8]

    mean_TAE_AC_X = TAE_AC_X / num
    mean_TAE_AC_Y = TAE_AC_Y / num
    mean_TAE_AC_Z = TAE_AC_Z / num
    mean_TAE_PC_X = TAE_PC_X / num
    mean_TAE_PC_Y = TAE_PC_Y / num
    mean_TAE_PC_Z = TAE_PC_Z / num
    mean_TAE_TC_X = TAE_TC_X / num
    mean_TAE_TC_Y = TAE_TC_Y / num
    mean_TAE_TC_Z = TAE_TC_Z / num

    return mean_TAE_AC_X, mean_TAE_AC_Y, mean_TAE_AC_Z, \
           mean_TAE_PC_X, mean_TAE_PC_Y, mean_TAE_PC_Z, \
           mean_TAE_TC_X, mean_TAE_TC_Y, mean_TAE_TC_Z


#ACPCTC绝对误差
def TAE(lab_batch, pred_batch):
    num = lab_batch.shape[0]
    batch_tre = np.abs(pred_batch - lab_batch)
    TRE_AC_X = 0.0
    TRE_AC_Y = 0.0
    TRE_AC_Z = 0.0
    TRE_PC_X = 0.0
    TRE_PC_Y = 0.0
    TRE_PC_Z = 0.0
    TRE_TC_X = 0.0
    TRE_TC_Y = 0.0
    TRE_TC_Z = 0.0
    for i in range(num):
        TRE_AC_X += batch_tre[i, 0]
        TRE_AC_Y += batch_tre[i, 1]
        TRE_AC_Z += batch_tre[i, 2]
        TRE_PC_X += batch_tre[i, 3]
        TRE_PC_Y += batch_tre[i, 4]
        TRE_PC_Z += batch_tre[i, 5]
        TRE_TC_X += batch_tre[i, 6]
        TRE_TC_Y += batch_tre[i, 7]
        TRE_TC_Z += batch_tre[i, 8]

    mean_TRE_AC_X = TRE_AC_X / num
    mean_TRE_AC_Y = TRE_AC_Y / num
    mean_TRE_AC_Z = TRE_AC_Z / num
    mean_TRE_PC_X = TRE_PC_X / num
    mean_TRE_PC_Y = TRE_PC_Y / num
    mean_TRE_PC_Z = TRE_PC_Z / num
    mean_TRE_TC_X = TRE_TC_X / num
    mean_TRE_TC_Y = TRE_TC_Y / num
    mean_TRE_TC_Z = TRE_TC_Z / num

    return mean_TRE_AC_X, mean_TRE_AC_Y, mean_TRE_AC_Z, \
           mean_TRE_PC_X, mean_TRE_PC_Y, mean_TRE_PC_Z,\
           mean_TRE_TC_X, mean_TRE_TC_Y, mean_TRE_TC_Z


def evaluation(truePoint, pred_Point):
    """Evaluate the quality of the predicted ACPC point at predicting the ACPC point.

    Args:
        CLEAR: The real ACPC point.
        pred_CLEAR: Predicted ACPC point by the model.

    Returns:
        mPSNR: mean PSNR between the real ACPC point and predicted ACPC point.
    """
    num = truePoint.shape[0]
    # mean_ace = 0.0
    # mean_pce = 0.0
    # for i in range(num):
    #     truep = truePoint
    #     pred_p = pred_Point
    #     mean_ace += math.sqrt((truep[0, 0] - pred_p[0, 0]) ** 2
    #                           + (truep[0, 1] - pred_p[0, 1]) ** 2
    #                           + (truep[0, 2] - pred_p[0, 2]) ** 2) / 256.0
    #
    #     mean_pce += math.sqrt((truep[0, 3] - pred_p[0, 3]) ** 2
    #                           + (truep[0, 4] - pred_p[0, 4]) ** 2
    #                           + (truep[0, 5] - pred_p[0, 5]) ** 2) / 256.0
    # mean_ace = mean_ace / num
    # mean_pce = mean_pce / num
    ACerror = np.square(truePoint[:, 0] - pred_Point[:, 0]) \
              + np.square(truePoint[:, 1] - pred_Point[:, 1]) \
              + np.square(truePoint[:, 2] - pred_Point[:, 2])
    ACm = 0.0
    for i in range(num):
        ACm += np.sqrt(ACerror[i])
    AC = ACm
    PCerror = np.square(truePoint[:, 3] - pred_Point[:, 3]) + \
              np.square(truePoint[:, 4] - pred_Point[:, 4]) + \
              np.square(truePoint[:, 5] - pred_Point[:, 5])
    PCm = 0.0
    for i in range(num):
        PCm += np.sqrt(PCerror[i])
    PC = PCm   #   / 256.0     这里除以的越小，误差越大。 是算ACPC距离误差占整幅图像的比例

    TCerror = np.square(truePoint[:,6] - pred_Point[:, 6]) + \
              np.square(truePoint[:,7] - pred_Point[:,7]) + \
              np.square(truePoint[:,8] - pred_Point[:,8])

    TCm = 0.0
    for i in range(num):
        TCm += np.sqrt(TCerror[i])
    TC = TCm
    mean_ace = AC / num
    mean_pce = PC / num
    mean_tce = TC / num
    # Attach a scalar summary to mPSNR
    # tf.summary.scalar("meanACerror", mean_ace)
    # tf.summary.scalar("meanPCerror", mean_pce)

    return mean_ace, mean_pce, mean_tce


# def thetaACPC(real_AP, pred_AP):
#     """
#         :param real_AP:
#         :param pred_AP:
#         :return: cal theta of the pred_ACPCline and the true_ACPCline
#         """
#
#     real_AP = tf.cast(real_AP, dtype=tf.float32)
#     pred_AP = tf.cast(pred_AP, dtype=tf.float32)
#     real_AC = tf.slice(real_AP, [0, 0], [BATCH_SIZE, 3])
#     real_PC = tf.slice(real_AP, [0, 3], [BATCH_SIZE, 3])
#     ACPCTrueline = tf.subtract(real_AC, real_PC)
#
#     pred_AC = tf.slice(pred_AP, [0, 0], [BATCH_SIZE, 3])
#     pred_PC = tf.slice(pred_AP, [0, 3], [BATCH_SIZE, 3])
#     ACPCPredline = tf.subtract(pred_AC, pred_PC)
#
#     True_norm = tf.sqrt(tf.reduce_sum(tf.square(ACPCTrueline), axis=1))
#     Pred_norm = tf.sqrt(tf.reduce_sum(tf.square(ACPCPredline), axis=1))
#     multisum = tf.reduce_sum(tf.multiply(ACPCTrueline, ACPCPredline), axis=1)
#     cosin = multisum / (True_norm * Pred_norm + 0.0001)
#     # cosin = tf.transpose(cosin, name="cosin")
#     theta = tf.acos(cosin, name="theta")
#     # theta = np.reshape(theta, [BATCH_SIZE, 1])
#     theta = tf.reduce_mean(theta)
#
#     # if cosin < 0:
#     #     theta = tf.subtract(tf.constant(pi, tf.float32), theta, name="theta")
#     return theta

def thetaACPC(real_AP, pred_AP):
    """
        :param real_AP:
        :param pred_AP:
        :return: cal theta of the pred_ACPCline and the true_ACPCline
        """
    real_AC = real_AP[:, 0:3]
    real_PC = real_AP[:, 3:6]
    ACPCTrueline = real_AC - real_PC

    pred_AC = pred_AP[:, 0:3]
    pred_PC = pred_AP[:, 3:6]
    ACPCPredline = pred_AC - pred_PC

    True_norm = np.sqrt(np.sum(np.square(ACPCTrueline), axis=1))
    Pred_norm = np.sqrt(np.sum(np.square(ACPCPredline), axis=1))
    multisum = np.sum(np.multiply(ACPCTrueline, ACPCPredline), axis=1)
    cosin = multisum / (True_norm * Pred_norm + 0.00001)
    cosin = np.mean(cosin)
    theta = math.acos(cosin)

    return theta

def restore_model(sess, saver, exist_model_dir, global_step):
    log_info = "Restoring Model From %s..." % exist_model_dir
    print(termcolor.colored(log_info, 'green', attrs=['bold']))
    ckpt = tf.train.get_checkpoint_state(exist_model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        init_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        sess.run(tf.assign(global_step, init_step))
    else:
        print('No Checkpoint File Found!')
        return

    return init_step

def train():
    global_step = tf.train.get_or_create_global_step()

    # Placeholders (IXI image:256*256*150pixels=9830400, label=6)
    Img_Batch = tf.placeholder(tf.float32, shape=[None, img_W, img_H, img_D, nChann])  # [None, 256*256*150]
    Lab_Batch = tf.placeholder(tf.float32, shape=[None, nLabel])  # [None, 9]
    print(termcolor.colored("Building Computation Graph...", 'green', attrs=['bold']))
    pred_model = md3.Model_3D(Img_Batch, True)
    train_loss = loss3d_ACPC.loss_l2(Lab_Batch, pred_model)
    #  #Attach a scalar summary to mse_loss
    tf.summary.scalar("loss", train_loss)

    train_op = optimize3d_ACPC.optimize(train_loss, learning_rate, global_step)
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)

    summ_op = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.log_device_placement = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # print(termcolor.colored("Defining Summary Writer...", 'green', attrs=['bold']))
    print("Defining Summary Writer...")
    summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

    step = 0
    if train_from_exist:
        step = restore_model(sess, saver, exist_model_dir, global_step)
    else:
        # print(termcolor.colored("Initializing Variables...", 'green', attrs = ['bold']))
        print("Initializing Variables...")
        sess.run(tf.global_variables_initializer())

    min_loss = float('Inf')
    minAC_error = float('Inf')
    minPC_error = float('Inf')
    minTC_error = float('Inf')

    min_tae_acx = float('Inf')
    min_tae_acy = float('Inf')
    min_tae_acz = float('Inf')
    min_tae_pcx = float('Inf')
    min_tae_pcy = float('Inf')
    min_tae_pcz = float('Inf')

    min_tae_tcx = float('Inf')
    min_tae_tcy = float('Inf')
    min_tae_tcz = float('Inf')

    train_name_list = os.listdir(data_dir + train_image_dir)
    num_names = len(train_name_list)
    index = np.arange(num_names / Batch_size)
    index = index * Batch_size

    # print(termcolor.colored("Starting To Train...", 'green', attrs=['bold'])) # original
    print("Starting To Train...")
    for i in range(EPOCH):
        for k in index:
            step += 1
            batch_train_img, batch_train_lab, _ = input_3DImage.get_data_MRI(data_dir, train_image_dir, train_lab_dir, Batch_size)
            feed_dict = {Img_Batch: batch_train_img, Lab_Batch: batch_train_lab}

            start_time = time.time()
            _, model_loss, pred_batch, summary_str = sess.run([train_op, train_loss, pred_model, summ_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)

            pred_batch = pred_batch.astype(np.int32)
            batch_train_lab = batch_train_lab.astype(np.int32)

            model_ACError, model_PCError, model_TCError = evaluation(batch_train_lab, pred_batch)
            TAE_AC_X, TAE_AC_Y, TAE_AC_Z, \
            TAE_PC_X, TAE_PC_Y, TAE_PC_Z,\
            TAE_TC_X, TAE_TC_Y, TAE_TC_Z = TAE(batch_train_lab, pred_batch)


            # 判断ACPC线预测是否都在5°以内
            angelACPCline = thetaACPC(batch_train_lab, pred_batch)
            angelACPCline = 180 / pi * angelACPCline

            # summary = tf.Summary()
            # summary_writer.add_summary(summary, step)

            duration = time.time() - start_time
            if (step + 1) % 1 == 0:
                examples_per_second = Batch_size / duration
                seconds_per_batch = float(duration)

                if model_loss < min_loss: min_loss = model_loss
                if model_ACError < minAC_error: minAC_error = model_ACError
                if model_PCError < minPC_error: minPC_error = model_PCError
                if model_TCError < minTC_error: minTC_error = model_TCError

                if TAE_AC_X < min_tae_acx: min_tae_acx = TAE_AC_X
                if TAE_AC_Y < min_tae_acy: min_tae_acy = TAE_AC_Y
                if TAE_AC_Z < min_tae_acz: min_tae_acz = TAE_AC_Z
                if TAE_PC_X < min_tae_pcx: min_tae_pcx = TAE_PC_X
                if TAE_PC_Y < min_tae_pcy: min_tae_pcy = TAE_PC_Y
                if TAE_PC_Z < min_tae_pcz: min_tae_pcz = TAE_PC_Z
                if TAE_TC_X < min_tae_tcx: min_tae_tcx = TAE_TC_X
                if TAE_TC_Y < min_tae_tcy: min_tae_tcy = TAE_TC_Y
                if TAE_TC_Z < min_tae_tcz: min_tae_tcz = TAE_TC_Z

                with open("Records/train_records.txt", "a") as file:
                    format_str = "%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n"
                    file.write(str(format_str) %
                               (step, model_loss, min_loss,
                                model_ACError, minAC_error,
                                model_PCError, minPC_error,
                                model_TCError, minTC_error,
                                examples_per_second, seconds_per_batch))

                # print(termcolor.colored('%s ---- step #%d' % (datetime.now(), step + 1), 'green', attrs=['bold']))
                print('%s ---- step #%d' % (datetime.now(), step))

                print("Pred: ")
                print(str(pred_batch.astype(np.int32)))
                print("True: ")
                print(str(batch_train_lab.astype(np.int32)))

                print('  LOSS = %.6f\t MIN_LOSS = %.6f' % (model_loss, min_loss))
                print('  model_ACError = %.6f\t minAC_error = %.6f' % (model_ACError, minAC_error))
                print('  model_PCError = %.6f\t minPC_error = %.6f' % (model_PCError, minPC_error))
                print('  model_TCError = %.6f\t minTC_error = %.6f' % (model_TCError, minTC_error))
                print('  TAE_AC_X = %.6f\t min_tae_acx = %.6f' % (TAE_AC_X, min_tae_acx))
                print('  TAE_AC_Y = %.6f\t min_tae_acy = %.6f' % (TAE_AC_Y, min_tae_acy))
                print('  TAE_AC_Z = %.6f\t min_tae_acz = %.6f' % (TAE_AC_Z, min_tae_acz))
                print('  TAE_PC_X = %.6f\t min_tae_pcx = %.6f' % (TAE_PC_X, min_tae_pcx))
                print('  TAE_PC_Y = %.6f\t min_tae_pcy = %.6f' % (TAE_PC_Y, min_tae_pcy))
                print('  TAE_PC_Z = %.6f\t min_tae_pcz = %.6f' % (TAE_PC_Z, min_tae_pcz))
                print('  TAE_TC_X = %.6f\t min_tae_tcx = %.6f' % (TAE_TC_X, min_tae_tcx))
                print('  TAE_TC_Y = %.6f\t min_tae_tcy = %.6f' % (TAE_TC_Y, min_tae_tcy))
                print('  TAE_TC_Z = %.6f\t min_tae_tcz = %.6f' % (TAE_TC_Z, min_tae_tcz))

                print('  ThetaLine = %.6f' % angelACPCline)
                # print('  ' + termcolor.colored(
                #     '%.6f images/sec\t%.6f seconds/batch' % (examples_per_second, seconds_per_batch), 'blue',
                #     attrs=['bold']))

                print(' %.6f images/sec\t%.6f seconds/batch' % (examples_per_second, seconds_per_batch))

            # if ((step + 1) % 200 == 0) or ((step + 1) == max_steps):
            #     summary_str = sess.run(summ_op, feed_dict=feed_dict)
            #     summary_writer.add_summary(summary_str, step + 1)

            if (step % 100 == 0) or (step == max_steps):
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                print("saving checkpoint into %s-%d" % (checkpoint_path, step))
                saver.save(sess, checkpoint_path, global_step=step)

            # 验证集
            if (step % 2 == 0) or (step == max_steps):
                val_data, val_label = input_3DImage.getvaliddata(valid_dir, valid_data, valid_label, 3)
                feed_dict = {Img_Batch: val_data}
                pred_batch = sess.run(pred_model, feed_dict=feed_dict)
                val_ACError, val_PCError, val_TCError = evaluation(val_label, pred_batch)

                # 判断ACPC线预测是否都在5°以内
                angelACPCline = thetaACPC(val_label, pred_batch)
                angelACPCline = 180 / pi * angelACPCline

                print("eval_ACerror: ", val_ACError)
                print("eval_PCerror: ", val_PCError)
                print("eval_TCerror: ", val_TCError)
                print("eval_angelACPCline ", angelACPCline)

                summary = tf.Summary()
                summary.value.add(tag='eval_ACerror', simple_value=val_ACError)
                summary.value.add(tag='eval_PCerror', simple_value=val_PCError)
                summary.value.add(tag='eval_TCerror', simple_value=val_TCError)
                summary_writer.add_summary(summary, step)

                valid_loss = np.mean(np.abs(val_label - pred_batch))

                with open("Records/valid_records.txt", "a") as file:
                    format_str = "%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n"
                    file.write(str(format_str) % (
                        step, valid_loss,
                        val_ACError, val_PCError, val_TCError,
                        angelACPCline,
                        (val_ACError+val_PCError+angelACPCline)/3))

    summary_writer.close()
    sess.close()


def main(argv=None):
    if not train_from_exist:
        if tf.gfile.Exists(train_log_dir):
            tf.gfile.DeleteRecursively(train_log_dir)
        tf.gfile.MakeDirs(train_log_dir)
    else:
        if not tf.gfile.Exists(exist_model_dir):
            raise ValueError("Train from existed model, but the target dir does not exist.")

        if not tf.gfile.Exists(train_log_dir):
            tf.gfile.MakeDirs(train_log_dir)
    train()


if __name__ == '__main__':
    tf.app.run()

