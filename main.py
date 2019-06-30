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
parser.add_argument('--hidden_neuron', type=int, default=100)
parser.add_argument('--lambda_value', type=float, default=1)

parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--train_ratio', type=float, default=0.9)
parser.add_argument('--batch_size', type=int, default=500)

parser.add_argument('--optimizer_method',
                    choices=['Adam', 'RMSProp'], default='Adam')
parser.add_argument('--grad_clip', type=bool, default=False)
parser.add_argument('--base_lr', type=float, default=0.0005)
parser.add_argument('--decay_epoch_step', type=int, default=25,
                    help="decay the learning rate for each n epochs")

parser.add_argument('--random_seed', type=int, default=1000)
parser.add_argument('--display_step', type=int, default=1)
parser.add_argument('--save_step', type=int, default=5)

args = parser.parse_args()
print("     ===> Args parsed\n")

# Random seeding
tf.set_random_seed(args.random_seed)
np.random.seed(args.random_seed)

# Detail about dataset
path = "data/intersect-20m"
data_name = 'intersect-20m'

# Allow tensorflow to grow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
print("     ===> Config set\n")

# Try to load Pre-processed data
try:
    filename = "{}/preprocessed_autorec_dataset".format(path)
    R, mask_R, train_R, train_mask_R, eval_R, eval_mask_R, \
    n_train_R, n_eval_R, \
    train_users_idx, train_items_idx, \
    eval_users_idx, eval_items_idx = pickle.load(open(filename, 'rb'))

    print("     ===> Preprocess data success\n")

# Preprocess again
except:

    print("     ===> Preprocessing data\n")

    R, mask_R, train_R, train_mask_R, eval_R, eval_mask_R, \
    n_train_R, n_eval_R, \
    train_users_idx, train_items_idx, \
    eval_users_idx, eval_items_idx = read_rating(path, args.train_ratio)

    # Theres issue with counting n_train, supposed to be number of nonzero relation - n_eval relation
    n_train_R = len(R.nonzero()[0]) - n_eval_R

    # Split preprocessed data by user (use 90% user for training, 10% user for testing)
    separator = int(R.shape[0] * args.train_ratio)

    test_R = R[separator:]
    test_mask_R = mask_R[separator:]
    test_train_R = train_R[separator:]
    test_train_mask_R = train_mask_R[separator:]
    test_eval_R = eval_R[separator:]
    test_eval_mask_R = eval_mask_R[separator:]

    R = R[:separator]
    mask_R = mask_R[:separator]
    train_R = train_R[:separator]
    train_mask_R = train_mask_R[:separator]
    eval_R = eval_R[:separator]
    eval_mask_R = eval_mask_R[:separator]

    # Set the number of eval and test
    n_test_eval_R = len(test_eval_R.nonzero()[0]) # non zero
    n_test_train_R = len(test_R.nonzero()[0]) - n_test_eval_R # non zero from R - non zero eval R

    # Save to file
    filename = "{}/preprocessed_autorec_dataset".format(path)
    pickle.dump((R, mask_R, train_R, train_mask_R, eval_R, eval_mask_R, n_train_R, n_eval_R, train_users_idx, train_items_idx, eval_users_idx, eval_items_idx), open(filename, 'wb'))

    filename = "{}/preprocessed_autorec_dataset_test".format(path)
    pickle.dump((test_R, test_mask_R, test_train_R, test_train_mask_R, test_eval_R, test_eval_mask_R, n_test_train_R, n_test_eval_R, train_users_idx, train_items_idx, eval_users_idx, eval_items_idx), open(filename, 'wb'))

    print("      ===> Preprocess data success\n")


num_users = R.shape[0]
num_items = R.shape[1]

with tf.Session(config=config) as sess:
    AutoRec = AutoRec(sess, args,
                      num_users, num_items,
                      R, mask_R, train_R, train_mask_R, eval_R, eval_mask_R,
                      n_train_R, n_eval_R,
                      train_users_idx, train_items_idx,
                      eval_users_idx, eval_items_idx)

    print("     ===> Running model...\n")

    AutoRec.run()
