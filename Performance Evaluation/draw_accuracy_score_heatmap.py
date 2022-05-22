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
    ax = fig.add_subplot(111)
    # ax.set_title('Accuracy Score vs Window size vs Length of the event sequence \n', fontsize=11)

    # masking to remove a certain portion from the heatmao
    # mask = np.zeros_like(dframe.corr())
    # mask[np.triu_indices_from(mask)] = True # if mask[i][j] == 0, it won't be removed

    # this loop is equivalent to "mask[np.triu_indices_from(mask)] = True"
    # for i in range(mask.shape[0]):
    #     for j in range(mask.shape[1]):
    #         if j>i:
    #             mask[i][j] = 1
    #         else:
    #             mask[i][j] = 0

    # seaborn.set(rc={'figure.figsize': (15, 10)})
    # bx = seaborn.heatmap(np.flip(fscore_box, 0), cmap="YlGnBu", ax=ax, xticklabels=wsize_box, yticklabels=np.flip(seqlen_box,0), square=False)
    # bx = seaborn.heatmap(fscore_box, cmap="YlGnBu", ax=ax, xticklabels=wsize_box, yticklabels=seqlen_box, square=False)
    # seaborn.heatmap(np.flip(fscore_box, 0).T, cmap="YlGnBu", xticklabels=np.flip(seqlen_box,0), yticklabels=wsize_box, square=False)
    seaborn.heatmap(fscore_box.T, cmap="YlGnBu", xticklabels=seqlen_box, yticklabels=wsize_box, square=False, annot=True)

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom=True, bottom=False, top=False, labeltop=False)

    # plt.xlabel('\n Sliding window size \n') # for first scenario
    # plt.ylabel('\n Length of the event sequence')

    plt.xlabel('Length of the event sequence \n', fontsize=12)  # for 2nd scenario
    plt.ylabel('Sliding window size', fontsize=12)

    # create a border
    # ax5.axhline(y=0, color='k', linewidth=1)
    # ax5.axhline(y=dframe.corr().shape[1], color='k', linewidth=1)
    # ax5.axvline(x=0, color='k', linewidth=1)
    # ax5.axvline(x=dframe.corr().shape[0], color='k', linewidth=1)

    plt.show()

