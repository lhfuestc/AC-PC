# -*- coding:utf-8 -*-
"""
Created on 2018/11/8 21:03

@ Author : zhl
"""
from __future__ import absolute_import
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import input_3DImage
import options3d_ACPC
import pandas as pd
import os
import math
import numpy as np
import model_se_3Dto2D as model
from datetime import datetime
import time
import termcolor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_dir = options3d_ACPC.DEFAULT_DATA_DIR
test_Img_dir = options3d_ACPC.DEFAULT_VALID_ACPC_DIR
test_Lab_dir = options3d_ACPC.DEFAULT_VALID_LABEL_DIR
img_H = options3d_ACPC.ImageH
img_W = options3d_ACPC.ImageW
img_D = options3d_ACPC.ImageD
nChann = options3d_ACPC.CHANNELS
nLabel = options3d_ACPC.InpCol

norm = options3d_ACPC.norm
pi = options3d_ACPC.PI
BATCH_SIZE = options3d_ACPC.params.batch_size

TEST_BATCH_SIZE = 1 # process a single image (NOT patch) one time
TEST_INTERVAL_SECS = 180 if options3d_ACPC.params.eval_once else options3d_ACPC.params.interval_secs
TOTAL_VALID_IMAGES = len(os.listdir(data_dir + test_Img_dir))

test_log_dir = options3d_ACPC.params.valid_log_dir
test_model_dir = options3d_ACPC.params.valid_model_dir
Label_save_dir = options3d_ACPC.params.TestResult_LABEL_dir
save_point = options3d_ACPC.params.save_ACPCpoint


def evaluationACPC(truePoint, pred_Point):
    """Evaluate the quality of the predicted CLEAR images at predicting the CLEAR iamges.

    Args:
        CLEAR: The real CLEAR images.
        pred_CLEAR: Predicted CLEAR images by the model.

    Returns:
        mPSNR: mean PSNR between the real CLEAR images and predicted CLEAR images.
    """
    mean_ace = 0.0
    mean_pce = 0.0
    for i in range(TEST_BATCH_SIZE):
        truep = truePoint
        pred_p = pred_Point
        mean_ace += math.sqrt((truep[0, 0] - pred_p[0, 0]) ** 2
                             +(truep[0, 1] - pred_p[0, 1]) ** 2
                             +(truep[0, 2] - pred_p[0, 2]) ** 2)

        mean_pce += math.sqrt((truep[0, 3]-pred_p[0, 3])**2
                             +(truep[0, 4]-pred_p[0, 4])**2
                             +(truep[0, 5]-pred_p[0, 5])**2)
    mean_ace = mean_ace/TEST_BATCH_SIZE
    mean_pce = mean_pce/TEST_BATCH_SIZE
    return mean_ace, mean_pce


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

def restore_model(sess, saver, model_dir):
    # Synchronous assessment!
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
        raise ValueError('no checkpoint file found!')

    return global_step

def testMain():
    #val_img, val_lab, filenames = input_1_3DImage.get_data_MRI(data_dir, test_Img_dir, test_Lab_dir, TOTAL_VALID_IMAGES)

    Img_Batch = tf.placeholder(dtype= tf.float32, shape=[None, img_W, img_H, img_D])
    print(termcolor.colored("Building Computation Graph...", 'green', attrs=['bold']))

    pred_model = model.moulti_cnn_lstm(Img_Batch, False)

    saver = tf.train.Saver()

    config = tf.ConfigProto(log_device_placement = True, allow_soft_placement = True)
    sess = tf.Session(config = config)

    print(termcolor.colored("Defining Summary Writer...", 'green', attrs=['bold']))
    summ_writer = tf.summary.FileWriter(test_log_dir, sess.graph)

    print(termcolor.colored("Restoring model ...", 'green', attrs=['bold']))
    global_step = restore_model(sess, saver, test_model_dir)

    print(termcolor.colored("The Total Training steps: %d" % int(global_step), 'green', attrs = ['bold']))

    img_index = 0
    num_images = 100

    # if num_images != num_labels:
    #     errstr = "image does not match label!"
    #     raise ValueError(errstr)

    Total_loss = 0.0
    Total_TAE_AC = 0.0
    Total_TAE_PC = 0.0
    Total_line = 0.0
    accounttrue = 0
    pccounttrue = 0
    testmodel_label = []
    thetalineTrue = 0


    min_ac_error = float('Inf')
    min_pc_error = float('Inf')
    # mintae_ac_error = float('Inf')
    # mintae_pc_error = float('Inf')
    min_loss = float('Inf')

    ACE = []
    PCE = []
    THETAE = []

    while img_index < num_images:
        val_img, val_lab, filenames = input_3DImage.get_data_MRI(data_dir, test_Img_dir, test_Lab_dir,
                                                                   2)
        #inp_img = val_img[0]
        #inp_lab = val_lab[0]

        start_time = time.time()
        out_lab = sess.run(pred_model, feed_dict={Img_Batch: val_img})
        duration = float(time.time() - start_time)

        #                取整存储
        a1 = int(round(out_lab[0, 0]))
        b1 = int(round(out_lab[0, 1]))
        c1 = int(round(out_lab[0, 2]))
        d1 = int(round(out_lab[0, 3]))
        e1 = int(round(out_lab[0, 4]))
        f1 = int(round(out_lab[0, 5]))
        value = [filenames[0], a1, b1, c1, d1, e1, f1]
        testmodel_label.append(value)

        model_loss = np.mean(np.abs(val_lab - out_lab))
        # TAE_AC, TAE_PC, = TAE(val_lab, out_lab)
        mean_ace, mean_pce = evaluationACPC(val_lab, out_lab)
        thetaACPCline = thetaACPC(val_lab, out_lab)
        thetaline = 180 / pi * thetaACPCline

        ACE.append(mean_ace)
        PCE.append(mean_pce)
        THETAE.append(thetaline)

        if mean_ace <= 7:     #mean_ace, mean_pce 应该等于TAE_AC， TAE_PC
            accounttrue += 1
        if mean_pce <= 7:
            pccounttrue += 1

        # 判断ACPC线预测是否都在5°以内
        if thetaline <= 7:
           thetalineTrue += 1

        a = round(val_lab[0, 0])
        b = round(val_lab[0, 1])
        c = round(val_lab[0, 2])
        d = round(val_lab[0, 3])
        e = round(val_lab[0, 4])
        f = round(val_lab[0, 5])

        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(val_img[0, :, :, c, 0], cmap='gray')
        plt.scatter(b, a, color='g', marker='+')  #AC真实
        plt.subplot(2, 2, 2)
        plt.imshow(val_img[0, :, :, f, 0], cmap='gray')
        plt.scatter(e, d, color='g', marker='+')  #PC真实
        plt.subplot(2, 2, 3)
        plt.imshow(val_img[0, :, :, c1, 0], cmap='gray')
        plt.scatter(b1, a1, color='r', marker='+')  # AC预测
        plt.subplot(2, 2, 4)
        plt.imshow(val_img[0, :, :, f1, 0], cmap='gray')
        plt.scatter(e1, d1, color='r', marker='+')  # PC预测
        plt.savefig("image/" + str(img_index) + ".png")
        # plt.show()


        summary = tf.Summary()
        summary.value.add(tag='Test_AC', simple_value = mean_ace)
        summary.value.add(tag='Test_PC', simple_value = mean_pce)
        summ_writer.add_summary(summary, img_index)

        if model_loss < min_loss: min_loss = model_loss
        if mean_ace < min_ac_error: min_ac_error = mean_ace
        if mean_pce < min_pc_error: min_pc_error = mean_pce
        print("Pred: ")
        print(str(val_lab.astype(np.int32)))
        print("True: ")
        print(str(out_lab.astype(np.int32)))


        print(termcolor.colored('%s ---- step #%d' % (datetime.now(), img_index + 1), 'green', attrs=['bold']))
        print('  Processing Image %d/%d...' % (img_index + 1, num_images))
        print('  LOSS = %.6f\t MIN_LOSS = %.6f' % (model_loss, min_loss))
        print('  acError = %.6f\t min_ac_error = %.6f' % (mean_ace, min_ac_error))
        print('  pcError = %.6f\t min_pc_error = %.6f' % (mean_pce, min_pc_error))
        print('  ThetaLine = %.6f' % thetaline)

        print('  ' + termcolor.colored('testing one image need %.6f seconds' % (duration), 'blue', attrs=['bold']))

        with open("test_records/test_records.txt", "a") as file:
            format_str = "%d\t%.6f\t%.6f\t%.6f\t%.6f\n"
            file.write(str(format_str) % (img_index + 1, model_loss, mean_ace, mean_pce, thetaline))

        # with open("test_records/test_records_WH.txt", "a") as file:
        #     format_str = "%d\t%s\t%.6f\n"
        #     file.write(str(format_str) % (img_index + 1, 'P_ACX', out_lab[0, 0]))
        #     file.write(str(format_str) % (img_index + 1, 'T_ACX', val_lab[0, 0]))
        #     file.write(str(format_str) % (img_index + 1, 'P_ACY', out_lab[0, 1]))
        #     file.write(str(format_str) % (img_index + 1, 'T_ACY', val_lab[0, 1]))
        #     file.write(str(format_str) % (img_index + 1, 'P_ACZ', out_lab[0, 2]))
        #     file.write(str(format_str) % (img_index + 1, 'T_ACZ', val_lab[0, 2]))
        #     file.write(str(format_str) % (img_index + 1, 'P_PCX', out_lab[0, 3]))
        #     file.write(str(format_str) % (img_index + 1, 'T_PCX', val_lab[0, 3]))
        #     file.write(str(format_str) % (img_index + 1, 'P_PCY', out_lab[0, 4]))
        #     file.write(str(format_str) % (img_index + 1, 'T_PCY', val_lab[0, 4]))
        #     file.write(str(format_str) % (img_index + 1, 'P_PCZ', out_lab[0, 5]))
        #     file.write(str(format_str) % (img_index + 1, 'T_PCZ', val_lab[0, 5]))

        Total_loss += model_loss
        Total_TAE_AC += mean_ace
        Total_TAE_PC += mean_pce
        Total_line += thetaline

        img_index += 1
    # column_name = ['filename', 'ACx', 'ACy', 'ACz', 'PCx', 'PCy', 'PCz']
    # xml_df = pd.DataFrame(testmodel_label, columns=column_name)
    # xml_df.to_csv(Label_save_dir, index=None)


    Mean_Loss = Total_loss / num_images
    Mean_TRE_AC = Total_TAE_AC / num_images
    Mean_TRE_PC = Total_TAE_PC / num_images
    Mean_line = Total_line / num_images
    ACaccuracy = accounttrue / num_images
    PCaccuracy = pccounttrue / num_images
    lineaccuracy = thetalineTrue / num_images

    print('  ' + termcolor.colored('Testing Process Done!', 'red', attrs=['bold']))
    print('  ' + termcolor.colored('Mean_AC_Error = %.6f\t Mean_PC_Error = %.6f' % (Mean_TRE_AC, Mean_TRE_PC), 'red',
                                   attrs=['bold']))
    print('  ' + termcolor.colored('Mean_line = %.6f' % Mean_line, 'red', attrs=['bold']))
    print('  ' + termcolor.colored('ACaccuracy = %.6f\t PCaccuracy = %.6f' % (ACaccuracy, PCaccuracy), 'red',
                                   attrs=['bold']))
    print('  ' + termcolor.colored('lineaccuracy = %.6f' % (lineaccuracy), 'red', attrs=['bold']))

    #方差

    #标准差
    print("AC_std = %.6f" % np.std(ACE))
    print("PC_std = %.6f" % np.std(PCE))
    print("linetheta_std = %.6f" % np.std(THETAE))

    print("AC_acc_5 = %.6f" % (sum(i <= 5 for i in ACE) / num_images))
    print("PC_acc_5 = %.6f" % (sum(i <= 5 for i in PCE) / num_images))
    print("linetheta_acc_5 = %.6f" % (sum(i <= 5 for i in THETAE) / num_images))
    print()

    print("AC_acc_6 = %.6f" % (sum(i <= 6 for i in ACE) / num_images))
    print("PC_acc_6 = %.6f" % (sum(i <= 6 for i in PCE) / num_images))
    print("linetheta_acc_6 = %.6f" % (sum(i <= 6 for i in THETAE) / num_images))

    print("AC_acc_10 = %.6f" % (sum(i <= 10 for i in ACE) / num_images))
    print("PC_acc_10 = %.6f" % (sum(i <= 10 for i in PCE) / num_images))
    print("linetheta_acc_10 = %.6f" % (sum(i <= 10 for i in THETAE) / num_images))


    with open("test_records/test_records_TFR.txt", "a") as file:
        format_str = "%d\t%.6f\t%.6f\t%.6f\t%.6f\n"
        file.write(format_str % (int(img_index), Mean_Loss, Mean_TRE_AC, Mean_TRE_PC, Mean_line))

    summ_writer.close()
    sess.close()

def main(argv = None):
    if tf.gfile.Exists(test_log_dir):
        tf.gfile.DeleteRecursively(test_log_dir)
    tf.gfile.MakeDirs(test_log_dir)
    testMain()


if __name__ == '__main__':
    tf.app.run()
