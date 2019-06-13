from data_preprocessor import *
from AutoRec import AutoRec
from datetime import datetime

import tensorflow as tf
import time
import argparse
import numpy as np
import pickle

current_time = time.time()

parser = argparse.ArgumentParser(description='I-AutoRec ')
parser.add_argument('--hidden_neuron', type=int, default=1000)
parser.add_argument('--lambda_value', type=float, default=1)

parser.add_argument('--train_epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=2048)

parser.add_argument('--optimizer_method',
                    choices=['Adam', 'RMSProp'], default='Adam')
parser.add_argument('--grad_clip', type=bool, default=False)
parser.add_argument('--base_lr', type=float, default=0.003)
parser.add_argument('--decay_epoch_step', type=int, default=50,
                    help="decay the learning rate for each n epochs")

parser.add_argument('--random_seed', type=int, default=1000)
parser.add_argument('--display_step', type=int, default=1)

args = parser.parse_args()
print("     ===> Args parsed\n")

# Random seeding
tf.set_random_seed(args.random_seed)
np.random.seed(args.random_seed)

# Detail about dataset
path = "data/intersect-20m"
data_name = 'intersect-20m'
num_users = 138493
num_items = 15085
num_total_ratings = 20000263
train_ratio = 0.9

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
print("     ===> Config set\n")

# Pre-processing data
try:
    filename = "{}/preprocessed_autorec_dataset".format(path)
    R, mask_R, train_R, train_mask_R, test_R, test_mask_R, \
    n_train_R, n_test_R, \
    train_users_idx, train_items_idx, \
    test_users_idx, test_items_idx = pickle.load(open(filename, 'rb'))

    print("     ===> Preprocess data success\n")

except:

    print("     ===> Preprocessing data\n")

    filename = "{}/preprocessed_autorec_dataset".format(path)
    R, mask_R, train_R, train_mask_R, test_R, test_mask_R, \
    n_train_R, n_test_R, \
    train_users_idx, train_items_idx, \
    test_users_idx, test_items_idx = read_rating(path, train_ratio)

    preprocessed_data = (R, mask_R, train_R, train_mask_R, test_R, test_mask_R,
                         n_train_R, n_test_R,
                         train_users_idx, train_items_idx,
                         test_users_idx, test_items_idx)
    pickle.dump(preprocessed_data, open(filename, 'wb'))

    print("      ===> Preprocess data success\n")


with tf.Session(config=config) as sess:
    AutoRec = AutoRec(sess, args,
                      num_users, num_items,
                      R, mask_R, train_R, train_mask_R, test_R, test_mask_R,
                      n_train_R, n_test_R,
                      train_users_idx, train_items_idx,
                      test_users_idx, test_items_idx)

    print("     ===> Running model...\n")

    AutoRec.run()
