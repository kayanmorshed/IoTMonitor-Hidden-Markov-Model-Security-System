#!/usr/bin/evn python

import numpy as np
import pandas as pd
import seaborn as sb
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from ipywidgets import interactive
from matplotlib import cm


if __name__ == "__main__":
    # read the datasets
    # dst = pd.read_csv('c1/fscore/fscore_wsize_seqlen_c1.csv')
    dst = pd.read_csv('c3/fscore/fscore_wsize_seqlen_c3.csv')
    # dst = pd.read_csv('c2_c4/c2_c4_only_fscores.csv') # for c2 and c4 combined

    # extract the columns
    fscore = dst['unordered_fscore'].values # for unordered fscore
    # fscore = dst['f-score'].values # for ordered fscore
    event_seq_len = dst['event_seq_len'].values
    window_size = dst['window_size'].values

    # create the grid
    # seq_len, wsize = np.meshgrid(event_seq_len, window_size)
    seq_len, wsize = np.meshgrid(event_seq_len, window_size)
    seq_len_flattened = seq_len.flatten()
    wsize_flattened = wsize.flatten()

    # set an empty 2d matrix for fscore
    fscore_2d = np.zeros((seq_len.shape[0], wsize.shape[1]))

    # loop through the grid to set corresponding fscores
    count = 0
    for i in range(0, seq_len.shape[0]):
        for j in range(0, wsize.shape[1]):
            # has to search in the dataset where seq_len[0][i] and wsize[j][0] matches and extract corresponding fscore

            current_seq_val = seq_len[0][i] # current value in the event_sequence_array: 12
            current_win_val = wsize[j][0] # current value in the window_size array: 5

            # get all indices where 'current_seq_val' present in 'event_seq_len' and
            # 'current_win_val' present in 'window_size'
            idx_arr_seq = np.where(event_seq_len == current_seq_val)[0]
            idx_arr_win = np.where(window_size == current_win_val)[0]

            # get the common index
            common_idx = np.intersect1d(idx_arr_seq, idx_arr_win)

            # set the value of 'fscore_2d[i][j]' as the value of 'fscore[common_idx]'
            if len(common_idx) < 1:
                continue
            else:
                # fscore_2d[i][j] = fscore[common_idx]
                fscore_2d[i][j] = fscore[common_idx[0]]

    # ****************************************************************************************
    # ************************* Start: Curve Fitting processing ******************************
    # ****************************************************************************************
    order = 2  # 1: linear, 2: quadratic

    if order == 1:
        # best-fit linear plane
        # A = np.c_[fscore_2d[:, 0], fscore_2d[:, 1], np.ones(fscore_2d.shape[0])]
        A = np.c_[fscore_2d[:, 0], fscore_2d[:, 1], np.zeros(fscore_2d.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, fscore_2d[:, 2])  # coefficients

        # evaluate it on grid
        Z = C[0] * seq_len + C[1] * wsize + C[2]

        # or expressed using matrix/vector product
        # Z = np.dot(np.c_[seq_len_flattened, wsize_flattened, np.ones(seq_len_flattened.shape)], C).reshape(wsize_flattened.shape)

    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(fscore_2d.shape[0]), fscore_2d[:, :2], np.prod(fscore_2d[:, :2], axis=1), fscore_2d[:, :2] ** 2]
        C, _, _, _ = scipy.linalg.lstsq(A, fscore_2d[:, 2])

        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(seq_len_flattened.shape), seq_len_flattened, wsize_flattened, seq_len_flattened * wsize_flattened, seq_len_flattened ** 2, wsize_flattened ** 2], C).reshape(seq_len.shape)

    # plot points and fitted surface
    fig = plt.figure()
    ax = fig.gca(projection='3d') # was in the original code
    # ax = fig.add_subplot(121, projection='3d')
    # cset = ax.plot_surface(seq_len, wsize, Z/10000, rstride=1, cstride=1, alpha=0.2) # was in the original code
    cset = ax.plot_surface(seq_len, wsize, Z, rstride=1, cstride=1, alpha=0.2)
    # cset = ax.plot_surface(seq_len, wsize, Z, color='gray', alpha=0.2)
    # ax.scatter(fscore_2d[:, 0], fscore_2d[:, 1], fscore_2d[:, 2], c='r') # was in the original code
    ax.scatter(seq_len, wsize, fscore_2d, c='r')
    plt.xlabel('Length of the Event Sequence')
    plt.ylabel('SLiding Window Size')
    ax.set_zlabel('Accuracy Score')

    # ax.set_zlim((0.0, 81.0))
    # plt.colorbar(cset)
    plt.show()


# *****************************************************************************************
# ***** original code from: https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6 *****
# *****************************************************************************************

# some 3-dim points
# mean = np.array([0.0, 0.0, 0.0])
# cov = np.array([[1.0, -0.5, 0.8], [-0.5, 1.1, 0.0], [0.8, 0.0, 1.0]])
# data = np.random.multivariate_normal(mean, cov, 50)
#
# # regular grid covering the domain of the data
# X, Y = np.meshgrid(np.arange(-3.0, 3.0, 0.5), np.arange(-3.0, 3.0, 0.5))
# XX = X.flatten()
# YY = Y.flatten()

# order = 1  # 1: linear, 2: quadratic
# if order == 1:
#     # best-fit linear plane
#     A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
#     C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients
#
#     # evaluate it on grid
#     Z = C[0] * X + C[1] * Y + C[2]
#
#     # or expressed using matrix/vector product
#     # Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)
#
# elif order == 2:
#     # best-fit quadratic curve
#     A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
#     C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
#
#     # evaluate it on a grid
#     Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)
#
# # plot points and fitted surface
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
# ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=50)
# plt.xlabel('X')
# plt.ylabel('Y')
# ax.set_zlabel('Z')
# ax.axis('equal')
# ax.axis('tight')
# plt.show()