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
    wdir = os.path.join(os.getcwd(), 'c1', 'sorted_combined') # for c1: undetectables and race conditions
    wdir_c3 = os.path.join(os.getcwd(), 'c3', 'sorted_combined') # for c3: no undetectables, but race conditions

    # get the filenames of all eval datasets
    file_list = [fname for fname in os.listdir(wdir) if '.csv' in fname]

    # sort the order of the files in fil_list
    file_list_sorted = []

    l = 5 # Set the starting length
    while l <= 20:
        file_name_to_add = 'eval_exec_combined_l' + str(l) + '_sorted.csv'
        file_list_sorted.append(file_name_to_add)
        l += 5 # define the distance between length of two consecutive files

    # draw the figure
    fig = plt.figure()

    # ***************** for c1 *****************
    # ax = fig.add_subplot(221)
    # ax.set_xlabel('Sliding Window Size', fontsize=11)
    # ax.set_ylabel('Parameter Estimating Time (in seconds)', fontsize=11)
    # ax.set_title('a) Estimating Time (with undetectable events)')
    #
    # ax2 = fig.add_subplot(222)
    # ax2.set_xlabel('Sliding Window Size', fontsize=11)
    # ax2.set_ylabel('Decoding Time (in seconds)', fontsize=11)
    # ax2.set_title('b) Decoding Time (with undetectable events)')

    # ***************** for c3 *****************
    ax3 = fig.add_subplot(221)  # for c3
    ax3.set_xlabel('Sliding Window Size', fontsize=12)
    ax3.set_ylabel('Probability Estimation Time (in milliseconds)', fontsize=12)
    # ax3.set_title('Probability estimating time vs Sliding window size \n vs Length of the event sequence \n')

    ax4 = fig.add_subplot(222) # for c3
    ax4.set_xlabel('Sliding Window Size', fontsize=12)
    ax4.set_ylabel('Decoding Time (in milliseconds)', fontsize=12)
    # ax4.set_title('Decoding time vs Sliding window size \n vs Length of the event sequence \n')

    color_list = ['red', 'green', 'blue', 'orange']
    color_counter = 0

    for item in file_list_sorted:
        # read the eval dataset
        dst = pd.read_csv(os.path.join(wdir, item))
        dst_c3 = pd.read_csv(os.path.join(wdir_c3, item)) # for c3

        # get the columns
        wsize = dst['window_size'].values
        wsize_c3 = dst_c3['window_size'].values # for c3

        # get seq len from dataset_name
        dst_name = dst['dataset_name'][0]
        search_length = re.search('l_(.*)_numerical', dst_name)  # for case C1. C3
        seqlen = int(search_length.group(1))  # window size in String -> int
        # seqlen = dst['event_seq_len'].values
        # seqlen_c3 = dst_c3['event_seq_len'].values # for c3

        est = dst['estimating_time'].values
        est_c3 = dst_c3['estimating_time'].values # for c3 (in seconds)
        est_c3_ms = est_c3 * 1000 # in milliseconds

        dcd = dst['decoding_time'].values
        dcd_c3 = dst_c3['decoding_time'].values # for c3 (in seconds)
        dcd_c3_ms = dcd_c3 * 1000 # in milliseconds

        # interpolation: wsize
        wsize_new = np.linspace(wsize.min(), wsize.max(), 500)
        wsize_c3_new = np.linspace(wsize_c3.min(), wsize_c3.max(), 500) # for c3

        # interpolate estimating time with window_new
        est_temp = make_interp_spline(wsize, est, k=3)
        est_c3_temp = make_interp_spline(wsize_c3, est_c3, k=3) # for c3
        dcd_temp = make_interp_spline(wsize, dcd, k=3)
        dcd_c3_temp = make_interp_spline(wsize_c3, dcd_c3, k=3) # for c3
        est_new = est_temp(wsize_new)
        est_c3_new = est_c3_temp(wsize_c3_new) # for c3
        dcd_new = dcd_temp(wsize_new)
        dcd_c3_new = dcd_c3_temp(wsize_c3_new) # for c3

        # choose hex for the randomly chosen color
        color = "%06x" % random.randint(0, 0xFF01EF)
        color2 = "%06x" % random.randint(0, 0xFF04FF)

        # label_ = 'length = ' + str(np.int64(seqlen[0])) # label for each line
        label_ = 'length = ' + str(np.int64(seqlen)) # label for each line
        # label_box.append('l-' + str(np.int64(seqlen[0]))) # for box plots

        # range of data to visualize
        start = 20
        end = 40

        # ************************** for c1 **************************
        # ax.plot(wsize[start:end], est[start:end], color='#' + color, marker='+', markersize=4, linewidth=1.5, label=label_) # without interpolation
        # ax2.plot(wsize[start:end], dcd[start:end], color='#' + color, marker='+', markersize=4, linewidth=1.5, label=label_) # without interpolation
        # ax.plot(wsize_new, est_new, color='#' + color, marker='+', markersize=2, linewidth=1.5, label=label_) # with interpolation
        # ax2.plot(wsize_new, dcd_new, color='#' + color, marker='+', markersize=2, linewidth=1.5, label=label_) # with interpolation

        # *********************** for c3 ****************************
        # in seconds
        # ax3.plot(wsize_c3[start:end], est_c3[start:end], color='#' + color, marker='o', markersize=4, linewidth=1.5, label=label_) # without interpolation
        # ax4.plot(wsize_c3[start:end], dcd_c3[start:end], color='#' + color, marker='o', markersize=4, linewidth=1.5, label=label_) # without interpolation

        # in milliseconds
        ax3.plot(wsize_c3[start:end], est_c3_ms[start:end], color=color_list[color_counter], marker='o', markersize=4, linewidth=2.0, label=label_)  # without interpolation
        ax4.plot(wsize_c3[start:end], dcd_c3_ms[start:end], color=color_list[color_counter], marker='o', markersize=4, linewidth=2.0, label=label_)  # without interpolation

        # ax3.plot(wsize_c3_new, est_c3_new, color='#' + color, marker='o', markersize=4, linewidth=1.5, label=label_)  # with interpolation
        # ax4.plot(wsize_c3_new, dcd_c3_new, color='#' + color, marker='o', markersize=4, linewidth=1.5, label=label_)  # with interpolation

        # Set the limit
        # plt.xlim(5, wsize.max())
        # plt.xlim(5, wsize_c3[10])
        plt.xlim(wsize[start], wsize[end-1])
        # plt.ylim(0.0)

        # increase color counter
        color_counter += 1

    # ****************************** for c1 ***********************
    # ax.legend()
    # # Show the major grid lines with dark grey lines
    # ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    # # Show the minor grid lines with very faint and almost transparent grey lines
    # ax.minorticks_on()
    # ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
    # # ax.show()
    #
    # ax2.legend()
    # # Show the major grid lines with dark grey lines
    # ax2.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    # # Show the minor grid lines with very faint and almost transparent grey lines
    # ax2.minorticks_on()
    # ax2.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)

    # ****************************** for c3 ***********************
    ax3.legend()
    # Show the major grid lines with dark grey lines
    ax3.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    # Show the minor grid lines with very faint and almost transparent grey lines
    ax3.minorticks_on()
    ax3.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)

    ax4.legend()
    # Show the major grid lines with dark grey lines
    ax4.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    # Show the minor grid lines with very faint and almost transparent grey lines
    ax4.minorticks_on()
    ax4.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)

    plt.tight_layout(h_pad=0.1)

    plt.show()
