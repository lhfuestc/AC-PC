# -*- coding:utf-8 -*-
"""
Created on 2018/11/3 16:18

@ Author : zhl
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import math
import os

parser = argparse.ArgumentParser()
#############################################################################################################
#                                          Global Constants                                                 #
#############################################################################################################

DEFAULT_DATA_DIR = '/home/zhanghuali/acpc/'
# DEFAULT_DATA_DIR = 'E:/deepLearning/tensorflow/graduation project/3D_ACPCLocation/ACPCChannelSE/'
DEFAULT_TRAIN_ACPC_DIR = 'data_v1111/trainACPCdata150/'
DEFAULT_TRAIN_LABEL_DIR = 'data_v1111/trainACPC150Label.txt'
DEFAULT_VALID_ACPC_DIR = 'data_v1111/testACPCdata150/'
DEFAULT_VALID_LABEL_DIR = 'data_v1111/testACPC150Label.txt'
DEFAULT_LABEL_SAVE_DIR = '/result/ResultOfTest.csv'
valid_dir = '/home/zhanghuali/acpc/'
valid_data = 'data_v1111/valid/'
valid_label = 'data_v1111/valid.txt'


SCALE = 1
ImageH, ImageW, ImageD = 256, 256, 150

InpRow = 1
InpCol = 6
CHANNELS = 1
norm = 10000.0   #所有IXI数据中像素最大值
# the Train maxvalue :  5587.68788767
# the Test maxvalue :  5370.70770264
PI = math.pi
lamd = 0.5 #0.5  # = 2 = 180 / PI

WEIGHT_DECAY = 0.0
TOTAL_LOSS_COLLECTION = 'my_total_losses'
##########################################################################################################
#                                               Data                                                     #
##########################################################################################################
parser.add_argument('--train_image_dir', type = str, default = DEFAULT_TRAIN_ACPC_DIR,
                    help = 'Path to IXI images for training.')
parser.add_argument('--train_label_dir', type = str, default = DEFAULT_TRAIN_LABEL_DIR,
                    help = 'Path to ACPC LABEL for training.')
parser.add_argument('--valid_image_dir', type = str, default = DEFAULT_VALID_ACPC_DIR,
                    help = 'Path to IXI images for validation.')
parser.add_argument('--valid_label_dir', type = str, default = DEFAULT_VALID_LABEL_DIR,
                    help = 'Path to ACPC LABEL for validation.')

##########################################################################################################
#                                               Train                                                    #
##########################################################################################################
parser.add_argument('--batch_size', type = int, default = 2,
                    help = 'Number of examples to process in a batch.')
parser.add_argument('--use_fp16', type = bool, default = False,
                    help = 'Train model using float16 data type.')
parser.add_argument('--xavier_init', type = bool, default = True,
                    help = 'whether to initialize params with Xavier method.')
parser.add_argument('--max_steps', type = int, default = 40000,
                    help = 'Maximum of the number of steps to train.')
parser.add_argument('--learning_rate', type = float, default = 1e-3,
                    help = 'The learning rate for optimizer.')
parser.add_argument('--train_log_freq', type = int, default = 100,
                    help = 'How often to log results to the console when training.')
parser.add_argument('--decay_steps', type = int, default = 1000,
                    help = 'How many steps to decay learning rate.')
parser.add_argument('--decay_rate', type = float, default = 0.9,
                    help = 'Decay rate for learning rate.')
parser.add_argument('--train_log_dir', type = str, default = 'train_logs/',
                    help = 'Directory where to write event logs and checkpoints.')
parser.add_argument('--normalization', type = bool, default = True,
                    help = 'Whether to normalize training examples.')
parser.add_argument('--sub_mean', type = bool, default = True,
                    help = 'Whether to subtract the average RGB values of the whole dataset.')
parser.add_argument('--train_from_exist', type = bool, default = True,
                    help = 'Whether to train model from pretrianed ones.')  #False
parser.add_argument('--exist_model_dir', type = str, default = 'train_logs/',
                    help = 'Directory where to load pretrianed models.')

##########################################################################################################
#                                                 Valid                                                  #
##########################################################################################################
parser.add_argument('--valid_log_dir', type = str, default = 'valid_logs/',
                    help = 'Directory where to write event logs and checkpoints.')
parser.add_argument('--eval_once', type = bool, default = False,
                    help = 'Whether to run eval only once.')
parser.add_argument('--interval_secs', type = int, default = 120,
                    help = 'How often to run the evalution.')
parser.add_argument('--TestResult_LABEL_dir', type = str, default = 'result/ResultOfTest.csv',
                    help = 'Directory where to save predicted ACPC point patches.')
parser.add_argument('--valid_model_dir', type = str, default = 'train_logs/',
                    help = 'Directory where the model that needs evaluation is saved.')
parser.add_argument('--luminance_only', type = bool, default = False,
                    help = 'Whether to evaluate PSNR and SSIM only over Luminance.')
parser.add_argument('--valid_subset_only', type = bool, default = True,
                    help = 'Whether to evaluate only a subset of the whole validation set (0801~0810).')
parser.add_argument('--save_ACPCpoint', type = bool, default = True,
                    help = 'Whether to save the result ACPC point when do evalution.')
parser.add_argument('--log_device_placement', type = bool, default = True,
                    help = 'Whether to log device placement.')

#%%
params = parser.parse_args()


