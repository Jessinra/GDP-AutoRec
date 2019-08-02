import argparse
import copy
import os
import pickle
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from AutoRec import AutoRec
from data_container import ProcessedDataContainer
from data_preprocessor import *

parser = argparse.ArgumentParser(description='I-Autoratingec ')
parser.add_argument('--hidden_neuron', type=int, default=100)
parser.add_argument('--lambda_value', type=float, default=1)
parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--train_ratio', type=float, default=0.9)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--optimizer_method', choices=['Adam', 'ratingMSProp'], default='Adam')
parser.add_argument('--grad_clip', type=bool, default=False)
parser.add_argument('--base_lr', type=float, default=0.0005)
parser.add_argument('--decay_epoch_step', type=int, default=25, help="decay the learning rate for each n epochs")
parser.add_argument('--random_seed', type=int, default=1000)
parser.add_argument('--display_step', type=int, default=1)
parser.add_argument('--save_step', type=int, default=5)

args = parser.parse_args()

# Allow tensorflow to grow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# ratingandom seeding
tf.set_random_seed(args.random_seed)
np.random.seed(args.random_seed)

# Detail about dataset
dataset_path = "data/intersect-20m"

if __name__ == "__main__":

    # Try to load Pre-processed data
    cached_dataset_train = "{}/preprocessed_autorec_dataset".format(dataset_path)
    cached_dataset_test = "{}/preprocessed_autorec_dataset_test".format(dataset_path)

    if os.path.exists(cached_dataset_train):
        print("loaded from cache : {}".format(cached_dataset_train))
        train_dataset_container = pickle.load(open(cached_dataset_train, 'rb'))

    else:
        train_dataset_container = get_rating_dataset_container(dataset_path, args.train_ratio)
        test_dataset_container = copy.deepcopy(train_dataset_container)

        # Split preprocessed data by user (use 90% user for training, 10% user for testing)
        separator = int(train_dataset_container.rating.shape[0] * args.train_ratio)
        train_dataset_container.slice(None, separator)
        test_dataset_container.slice(separator, None)

        print("saved to cache : {}".format(cached_dataset_train))
        pickle.dump(train_dataset_container, open(cached_dataset_train, 'wb'))

        print("saved to cache : {}".format(cached_dataset_test))
        pickle.dump(test_dataset_container, open(cached_dataset_test, 'wb'))

    with tf.Session(config=config) as sess:
        AutoRec = AutoRec(sess, args, train_dataset_container)
        AutoRec.run()
