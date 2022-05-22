import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from ipywidgets import interactive
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


if __name__ == "__main__":
    # read the datasets
    # dst = pd.read_csv('c3/fscore/fscore_wsize_seqlen_c1.csv')
    dst = pd.read_csv('c3/fscore/fscore_wsize_seqlen_c3.csv')
    # dst = pd.read_csv('c2_c4/c2_c4_only_fscores.csv') # for c2 and c4 combined

    # extract the columns
    fscore = dst['unordered_fscore'].values # for unordered fscore
    # fscore = dst['f-score'].values # for ordered fscore
    event_seq_len = dst['event_seq_len'].values
    window_size = dst['window_size'].values

    # create the grid
    seq_len, wsize = np.meshgrid(event_seq_len, window_size)

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

    # **************************************************************************
    # ********************** plot 3d figures ***********************************
    # **************************************************************************

    # ***************************** plot: heatmap (seaborn) ******************************
    # fig = plt.figure()
    # ax = fig.add_subplot(121)
    # sb.heatmap(fscore_2d, vmin=0.0, vmax=1.0, cmap='gray') # use it for heatmap by seaborn
    # ax.set_xlabel('Sequence Length')
    # ax.set_ylabel('Window Size')
    # ax.set_zlabel('Fscore')
    # plt.savefig('heatmap_c1.png')
    # plt.show()

    # ***************************** plot: contourf ***************************************
    # plot heatmap using contourf
    fig = plt.figure()
    ax2 = fig.add_subplot(121, projection='3d')
    cset = ax2.contourf(seq_len, wsize, fscore_2d, 200, cmap='winter') # for unordered fscore
    # cset = ax2.contourf(seq_len, wsize, fscore_2d, 200, cmap=cm.BrBG) # for ordered fscore
    ax2.set_zlim((0.0, 1.0))
    plt.colorbar(cset)

    # ax2.set_title("F-score (Order not considered) vs Window Size vs Length of the Event Sequence \n\n\n") # for unordered fscore
    ax2.set_title("Accuracy Score vs Window Size vs Length of the Event Sequence \n\n\n") # for ordered fscore
    ax2.set_xlabel('\n Length of the Event Sequence')
    ax2.set_ylabel('\n Window Size')
    # ax2.set_zlabel('F-score')
    ax2.set_zlabel('\n F-score (Functional dependencies relaxed)')

    # fig.text(.5, .05, 'Relationship among accuracy score, window size, and length of the event sequence', ha='center')

    # plt.savefig('contour_c1.png')

    plt.show()

    # ax2 = fig.add_subplot(122, projection='3d')
    # cset = ax2.contour3D(seq_len, wsize, fscore_2d)
    #
    # ax2.set_xlabel('Sequence Length')
    # ax2.set_ylabel('Window Size')
    # ax2.set_zlabel('Fscore')

    # plt.show()





