from data_preprocessor import *
from AutoRec import AutoRec

from ipywidgets import FloatProgress, IntProgress
from IPython.display import display
from datetime import datetime

import tensorflow as tf
import time
import argparse
import numpy as np
import pandas as pd
import pickle

current_time = time.time()

parser = argparse.ArgumentParser(description='I-AutoRec ')
parser.add_argument('--hidden_neuron', type=int, default=500)
parser.add_argument('--lambda_value', type=float, default=1)

parser.add_argument('--train_epoch', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=200)

parser.add_argument('--optimizer_method',
                    choices=['Adam', 'RMSProp'], default='Adam')
parser.add_argument('--grad_clip', type=bool, default=False)
parser.add_argument('--base_lr', type=float, default=1e-3)
parser.add_argument('--decay_epoch_step', type=int, default=50,
                    help="decay the learning rate for each n epochs")

parser.add_argument('--random_seed', type=int, default=1000)
parser.add_argument('--display_step', type=int, default=1)

args = parser.parse_args()
print("\n ===> Args parsed\n")

# Random seedings
tf.set_random_seed(args.random_seed)
np.random.seed(args.random_seed)

# Detail about dataset
data_name = 'ml-20m'
num_users = 138493
num_users = 1384

num_items = 26744
num_total_ratings = 20000263
train_ratio = 0.9

path = "data/{}/".format(data_name)
result_path = 'results/{}/{}_{}_{}_{}/'.format(data_name,
                                                str(args.random_seed),
                                                str(args.optimizer_method),
                                                str(args.base_lr),
                                                str(current_time))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
print("\n ===> Config set\n")

# Preprocessing data
try:
    filename = "preprocessed/ml-20m.1559115930.037085"
    R, mask_R, train_R, train_mask_R, test_R, test_mask_R,\
        n_train_R, n_test_R,\
        train_users_idx, train_items_idx,\
        test_users_idx, test_items_idx = pickle.load(open(filename, 'rb'))

    print("\n ===> Preprocess data success\n")

except:

    print("\n ===> Preprocessing data\n")

    R, mask_R, train_R, train_mask_R, test_R, test_mask_R,\
        n_train_R, n_test_R,\
        train_users_idx, train_items_idx,\
        test_users_idx, test_items_idx = read_rating(path, train_ratio)

    now = datetime.now()
    timestamp = datetime.timestamp(now)
    filename = "preprocessed/ml-20m.{}".format(str(timestamp))

    preprocessed_data = (R, mask_R, train_R, train_mask_R, test_R, test_mask_R,
                            n_train_R, n_test_R,
                            train_users_idx, train_items_idx,
                            test_users_idx, test_items_idx)

    print("\n ===> Preprocess data success\n")

    pickle.dump(preprocessed_data, open(filename, 'wb'))

with tf.Session(config=config) as sess:
    AutoRec = AutoRec(sess, args,
                        num_users, num_items,
                        R, mask_R, train_R, train_mask_R, test_R, test_mask_R,
                        n_train_R, n_test_R,
                        train_users_idx, train_items_idx,
                        test_users_idx, test_items_idx,
                        result_path)

    print("\n ===> Running model...\n")                    
    AutoRec.run()
