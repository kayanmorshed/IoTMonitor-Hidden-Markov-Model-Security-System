import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import os
import re
import seaborn
import random
from scipy.interpolate import make_interp_spline


if __name__ == '__main__':
    # set the working directory (of the 'sorted' folder)
    # wdir = os.path.join(os.getcwd(), 'c1', 'sorted') # for c1: undetectables and race conditions
    wdir = os.path.join(os.getcwd(), 'c3_aug20', 'sorted') # for c3: no undetectables, but race conditions

    # get the filenames of all eval datasets
    file_list = [fname for fname in os.listdir(wdir) if '.csv' in fname]

    # sort the order of the files in fil_list
    file_list_sorted = []
    l = 2 # Set the starting length
    while l <= 30:
        file_name_to_add = 'performance_eval_l' + str(l) + '_sorted.csv'
        file_list_sorted.append(file_name_to_add)
        l += 1 # define the distance between length of two consecutive files

    # define srating window, ending window and num_of_seqlen
    start_window = 19
    end_window = 40
    total_seqlen = 29

    # define variables to hold the lists 'wsize', 'seqlen', 'fscore'
    wsize_box = np.zeros(end_window-start_window, dtype=int) # window range = end window - start window
    seqlen_box = np.zeros(total_seqlen, dtype=int)
    fscore_box = np.zeros((total_seqlen, end_window-start_window))

    loop_counter = 1

    for item in file_list_sorted:
        # read the eval dataset
        dst = pd.read_csv(os.path.join(wdir, item))

        # get window column
        wsize = dst['window_size'].values[start_window:end_window]
        if wsize_box[0] == 0 and wsize_box[20] == 0:
            for i in range(0, end_window-start_window):
                wsize_box[i] = wsize[i].astype(np.int64)


        # get seq len from dataset_name
        search_length = re.search('l_(.*)_numerical', dst['dataset_name'][start_window])  # for case C1. C3
        seqlen = int(search_length.group(1))  # window size in String -> int
        seqlen_box[loop_counter-1] = seqlen

        # get fscore
        fscore = dst['unordered_fscore'].values[start_window:end_window] # take from window 100 to 200
        j = 0
        while j < end_window-start_window:
            fscore_box[loop_counter-1][j] = fscore[j]
            j += 1

        loop_counter += 1

    # ***************************************************************************************
    # ********************** For Correlation Heatmap using Seaborn **************************
    # ***************************************************************************************

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
    cset = ax.contourf(seqlen_box, wsize_box, fscore_box, cmap='winter')  # for unordered fscore
    # cset = ax2.contourf(seq_len, wsize, fscore_2d, 200, cmap=cm.BrBG) # for ordered fscore
    ax.set_zlim((0.0, 1.0))
    plt.colorbar(cset)

    # ax2.set_title("F-score (Order not considered) vs Window Size vs Length of the Event Sequence \n\n\n") # for unordered fscore
    ax.set_title("Accuracy Score vs Window Size vs Length of the Event Sequence \n\n\n")  # for ordered fscore
    ax.set_xlabel('\n Length of the Event Sequence')
    ax.set_ylabel('\n Window Size')
    # ax2.set_zlabel('F-score')
    ax.set_zlabel('\n Accuracy Score')

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom=True, bottom=False, top=False, labeltop=False)

    # plt.xlabel('\n Sliding window size \n') # for first scenario
    # plt.ylabel('\n Length of the event sequence')

    # plt.xlabel('Length of the event sequence \n', fontsize=12)  # for 2nd scenario
    # plt.ylabel('Sliding window size', fontsize=12)

    # create a border
    # ax5.axhline(y=0, color='k', linewidth=1)
    # ax5.axhline(y=dframe.corr().shape[1], color='k', linewidth=1)
    # ax5.axvline(x=0, color='k', linewidth=1)
    # ax5.axvline(x=dframe.corr().shape[0], color='k', linewidth=1)

    plt.show()

