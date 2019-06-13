import numpy as np
import pandas as pd
import pickle
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix, csr_matrix


def read_rating(path, train_ratio):
    
    filename = path + "/ratings.csr"
    R = pickle.load(open(filename, 'rb'))

    train_R, test_R, n_train_R, n_test_R, train_users_idx, train_items_idx, test_users_idx, test_items_idx = train_test_split(
        R, train_ratio=train_ratio, random_state=55)

    mask_R = mask_if_not_zero(R)
    train_mask_R = mask_if_not_zero(train_R)
    test_mask_R = mask_if_not_zero(test_R)

    return R, mask_R, train_R, train_mask_R, test_R, test_mask_R, n_train_R, n_test_R, train_users_idx, train_items_idx, test_users_idx, test_items_idx


def train_test_split(data, train_ratio, random_state=55):
    np.random.seed(random_state)

    # Initialize
    train = data.copy()
    test = lil_matrix(data.shape)
    test_users_idx = set()
    test_items_idx = set()

    # Usable rating for train / test
    nonzero = data.nonzero()
    n_nonzero = len(nonzero[0])

    n_train = int(train_ratio * n_nonzero)
    n_test = n_nonzero - n_train

    sampled_idx = np.random.choice(
        np.arange(n_nonzero), size=n_test, replace=False)

    # Create train set
    train_users_idx = set(np.arange(data.shape[0]))
    train_items_idx = set(np.arange(data.shape[1]))

    for idx in sampled_idx:
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


def mask_if_not_zero(matrix):
    nonzero_idx = matrix.nonzero()
    keep = np.arange(len(nonzero_idx[0]))
    n_keep = len(keep)

    mask_csr = csr_matrix(
        (np.ones(n_keep), (nonzero_idx[0][keep], nonzero_idx[1][keep])), shape=matrix.shape)
    return mask_csr
