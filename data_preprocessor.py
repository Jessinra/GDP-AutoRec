import pickle

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm

from data_container import ProcessedDataContainer


def get_rating_dataset_container(dataset_path, train_ratio):

    rating_filename = dataset_path + "/ratings.csr"
    cached_csr_rating = pickle.load(open(rating_filename, 'rb'))

    container = ProcessedDataContainer()


    train_rating, test_rating,\
        n_train_rating, n_test_rating, \
        train_users_idx, train_items_idx, \
        test_users_idx, test_items_idx = _train_test_split(cached_csr_rating, train_ratio=train_ratio, random_state=55)

    # Wrap things into single object
    container.rating = cached_csr_rating
    container.mask_rating = _mask_if_not_zero(cached_csr_rating)

    container.train_rating = train_rating
    container.train_mask_rating = _mask_if_not_zero(train_rating)

    container.test_rating = test_rating
    container.test_mask_rating = _mask_if_not_zero(test_rating)

    container.n_train_rating = n_train_rating
    container.n_test_rating = n_test_rating

    container.train_users_idx = train_users_idx
    container.train_items_idx = train_items_idx

    container.test_users_idx = test_users_idx
    container.test_items_idx = test_items_idx

    return container


def _train_test_split(rating_dataset, train_ratio, random_state=55):
    np.random.seed(random_state)

    # Initialize
    train = rating_dataset.copy()
    test = lil_matrix(rating_dataset.shape)
    test_users_idx = set()
    test_items_idx = set()

    # Usable rating for train / test
    nonzero = rating_dataset.nonzero()
    n_nonzero = len(nonzero[0])

    n_train = int(train_ratio * n_nonzero)
    n_test = n_nonzero - n_train

    sampled_idx = np.random.choice(np.arange(n_nonzero), size=n_test, replace=False)

    # Create train set
    train_users_idx = set(np.arange(rating_dataset.shape[0]))
    train_items_idx = set(np.arange(rating_dataset.shape[1]))

    for idx in tqdm(sampled_idx):

        row = nonzero[0][idx]
        col = nonzero[1][idx]

        # Modify matrix
        test[row, col] = train[row, col]
        train[row, col] = 0

        # Add into test set
        test_users_idx.add(row)
        test_items_idx.add(col)

    return train, test, n_train, n_test, \
        train_users_idx, train_items_idx, \
        test_users_idx, test_items_idx


def _mask_if_not_zero(matrix):
    
    nonzero_idx = matrix.nonzero()
    keep = np.arange(len(nonzero_idx[0]))
    n_keep = len(keep)

    return csr_matrix((np.ones(n_keep), (nonzero_idx[0][keep], nonzero_idx[1][keep])), shape=matrix.shape)
