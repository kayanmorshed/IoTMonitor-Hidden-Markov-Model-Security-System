#!/usr/bin/evn python

import numpy as np
import pandas as pd
import os
import seaborn as sb
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from ipywidgets import interactive
from matplotlib import cm


if __name__ == "__main__":
    # set the working directory (of the 'sorted' folder)
    wdir_c3 = os.path.join(os.getcwd(), 'c3')  # for c3: no undetectables, but race conditions

    # read the eval dataset
    dst = pd.read_csv(os.path.join(wdir_c3, 'all_lengths_timestep_conv_min.csv'))

    # get the columns
    num_states = dst['num_states']
    num_observ = dst['num_observ']
    timestep_conv = dst['timestep_conv']

    # create the grid
    states, observations = np.meshgrid(num_states, num_observ)
    states_flattened = states.flatten()
    observations_flattened = observations.flatten()

    # set an empty 2d matrix for timestep
    timestep_space = np.zeros((states.shape[0], observations.shape[1]))

    # loop through the grid to set corresponding fscores
    count = 0
    for i in range(0, states.shape[0]):
        for j in range(0, observations.shape[1]):
            # has to search in the dataset where states[0][i] and observations[j][0] matches
            # and extract corresponding timestep
            current_state = states[0][i] # current value in the event_sequence_array: 12
            current_obsrv = observations[j][0] # current value in the window_size array: 5

            # get all indices where 'current_state' present in 'states' and
            # 'current_obsrv' present in 'observations'
            idx_arr_state = np.where(current_state == states)[0]
            idx_arr_obsrv = np.where(current_obsrv == observations)[0]

            # get the common index
            common_idx = np.intersect1d(idx_arr_state, idx_arr_obsrv)

            # set the value of 'timestep_space[i][j]' as the value of 'timestep_conv[common_idx]'
            if len(common_idx) < 1:
                continue
            else:
                # fscore_2d[i][j] = fscore[common_idx]
                timestep_space[i][j] = timestep_conv[common_idx[0]]

    # ****************************************************************************************
    # ************************* Start: Curve Fitting processing ******************************
    # ****************************************************************************************
    order = 1  # 1: linear, 2: quadratic

    if order == 1:
        # best-fit linear plane
        # A = np.c_[fscore_2d[:, 0], fscore_2d[:, 1], np.ones(fscore_2d.shape[0])]
        A = np.c_[timestep_space[:, 0], timestep_space[:, 1], np.zeros(timestep_space.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, timestep_space[:, 2])  # coefficients

        # evaluate it on grid
        Z = C[0] * states + C[1] * observations + C[2]

        # or expressed using matrix/vector product
        # Z = np.dot(np.c_[states_flattened, observations_flattened, np.ones(states_flattened.shape)], C).reshape(observations_flattened.shape)

    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(timestep_space.shape[0]), timestep_space[:, :2], np.prod(timestep_space[:, :2], axis=1), timestep_space[:, :2] ** 2]
        C, _, _, _ = scipy.linalg.lstsq(A, timestep_space[:, 2])

        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(states_flattened.shape), states_flattened, observations_flattened, states_flattened * observations_flattened, states_flattened ** 2, observations_flattened ** 2], C).reshape(states.shape)

    # plot points and fitted surface
    fig = plt.figure()
    ax = fig.gca(projection='3d') # was in the original code
    # ax = fig.add_subplot(121, projection='3d')
    # cset = ax.plot_surface(seq_len, wsize, Z/10000, rstride=1, cstride=1, alpha=0.2) # was in the original code

    cset = ax.plot_surface(states, observations, Z, rstride=1, cstride=1, alpha=0.2) # our code

    # cset = ax.plot_surface(seq_len, wsize, Z, color='gray', alpha=0.2)
    # ax.scatter(fscore_2d[:, 0], fscore_2d[:, 1], fscore_2d[:, 2], c='r') # was in the original code
    ax.scatter(states, observations, timestep_space, c='r')

    plt.xlabel('X = Number of unique true states')
    plt.ylabel('Y = Number of unique observation states')
    ax.set_zlabel('Z = Timestep to converge')

    # ax.set_zlim((0.0, 81.0))
    plt.colorbar(cset)
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