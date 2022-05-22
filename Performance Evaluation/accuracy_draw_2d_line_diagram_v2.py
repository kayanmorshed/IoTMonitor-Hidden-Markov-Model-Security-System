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
    wdir = os.path.join(os.getcwd(), 'c3', 'sorted_combined') # for c3: no undetectables, but race conditions

    # get the filenames of all eval datasets
    file_list = [fname for fname in os.listdir(wdir) if '.csv' in fname]

    # sort the order of the files in fil_list
    file_list_sorted = []
    l = 5 # Set the starting length
    while l <= 20:
        file_name_to_add = 'eval_exec_combined_l' + str(l) + '_sorted.csv'
        file_list_sorted.append(file_name_to_add)
        l += 5 # define the distance between length of two consecutive files

    # set a varibale to hold fscore for box plots
    fscore_box = []
    label_box = []

    # define a dataFrame to draw the correlation heatmap
    dframe = pd.DataFrame()

    # draw the figure
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_xlabel('Sliding Window Size', fontsize=11)
    ax.set_ylabel('Accuracy Score', fontsize=11)
    ax.set_title('a) Accuracy Score (sequence with no undetectable events)')

    for item in file_list_sorted:
        # read the eval dataset
        dst = pd.read_csv(os.path.join(wdir, item))

        # get the columns
        wsize = dst['window_size'].values

        # get seq len from dataset_name
        dst_name = dst['dataset_name'][0]
        search_length = re.search('l_(.*)_numerical', dst_name)  # for case C1. C3
        seqlen = int(search_length.group(1))  # window size in String -> int
        # seqlen = dst['event_seq_len'].values

        # fscore = dst['f-score'].values
        # unordered_fscore = dst['f-score'].values
        unordered_fscore = dst['unordered_fscore'].values
        fscore_box.append(unordered_fscore) # for box plot

        # add 'wsize' into 'window_size column of the dataframe'
        # if not dframe['window_size'].values:
        #     dframe['window_size'] = wsize
        # dframe['window_size'] = wsize
        # also add 'unordered_fscore' into DataFrame
        # fscore_idx = 'sequence_length_' + str(seqlen[0])
        fscore_idx = 'sequence_length_' + str(seqlen)
        dframe[fscore_idx] = unordered_fscore

        # interpolation: wsize
        wsize_new = np.linspace(wsize.min(), wsize.max(), 400)

        # interpolate precisions with window_new
        precision_temp = make_interp_spline(wsize, unordered_fscore, k=3)
        precision_new = precision_temp(wsize_new)

        # choose hex for the randomly chosen color
        color = "%06x" % random.randint(0, 0xFF01FF)

        # label_ = 'length = ' + str(np.int64(seqlen[0])) # label for each line
        label_ = 'length = ' + str(np.int64(seqlen)) # label for each line
        # label_box.append('length = ' + str(np.int64(seqlen[0]))) # for box plots
        label_box.append('length = ' + str(np.int64(seqlen))) # for box plots

        ax.plot(wsize, unordered_fscore, color='#' + color, linewidth=1.5, label=label_) # without interpolation
        # ax.plot(wsize_new, precision_new, color='#' + color, linewidth=2, label=label_) # with interpolated data

        # Set the limit
        plt.xlim(2, wsize.max())
        plt.ylim(0.2, 1.01)

    ax.legend(loc="lower right")
    # Show the major grid lines with dark grey lines
    ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    # Show the minor grid lines with very faint and almost transparent grey lines
    ax.minorticks_on()
    ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)

    # draw boxplot
    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('Length of the event sequence', fontsize=11)
    ax2.set_ylabel('Accuracy Score', fontsize=11)
    ax2.set_title('b) Boxplot: Accuracy score vs Length of the event sequence \n')

    # draw the boxplot
    ax2.boxplot(fscore_box, labels=label_box)
    # Show the major grid lines with dark grey lines
    # ax2.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    # Show the minor grid lines with very faint and almost transparent grey lines
    # ax2.minorticks_on()
    # ax2.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)

    plt.show()

    # draw the figure
    # fig_box = plt.figure()
    # ax2 = fig_box.add_subplot(111)
    #
    # bp = ax2.boxplot(fscore_box, labels=label_box)
    #
    # plt.show()

    # ***************************************************************************************
    # ********************** For Correlation Heatmap using Seaborn **************************
    # ***************************************************************************************
    fig = plt.figure()
    ax5 = fig.add_subplot(111)

    ax5.set_title('Correlation between accuracy scores determined for different sequence length \n')

    # plt.figure(figsize=(5, 5))
    xlabel_coord = ['length = 5', 'length = 10', 'length = 15', 'length = 20']
    ylabel_coord = ['length = 5', 'length = 10', 'length = 15', 'length = 20']

    # masking to remove a certain portion from the heatmao
    mask = np.zeros_like(dframe.corr())
    # mask[np.triu_indices_from(mask)] = True # if mask[i][j] == 0, it won't be removed

    # this loop is equivalent to "mask[np.triu_indices_from(mask)] = True"
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if j>i:
                mask[i][j] = 1
            else:
                mask[i][j] = 0

    print(mask)

    seaborn.heatmap(dframe.corr(), annot=True, cmap="YlGnBu", ax=ax5, xticklabels=xlabel_coord, yticklabels=ylabel_coord, square=True)
    # seaborn.heatmap(dframe.corr(), annot=True, cmap="YlGnBu", ax=ax5, xticklabels=xlabel_coord, yticklabels=ylabel_coord, mask=mask, vmax=1.0, square=True)
    plt.xticks(rotation=0)

    # create a border
    ax5.axhline(y=0, color='k', linewidth=1)
    ax5.axhline(y=dframe.corr().shape[1], color='k', linewidth=1)
    ax5.axvline(x=0, color='k', linewidth=1)
    ax5.axvline(x=dframe.corr().shape[0], color='k', linewidth=1)

    plt.show()

