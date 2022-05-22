import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn
from scipy.interpolate import make_interp_spline


if __name__ == '__main__':
    # read the eval datasets
    # dst = pd.read_csv('c3/fscore/fscore_wsize_seqlen_c3.csv')
    dst = pd.read_csv('c1/fscore/fscore_wsize_seqlen_c1.csv')

    # get the columns
    wsize = dst['window_size'].values
    seqlen = dst['event_seq_len'].values
    fscore = dst['f-score'].values
    unordered_fscore = dst['unordered_fscore'].values

    print(wsize[0:40])
    print(fscore[0:40])

    current_seqlen = 10 # start from 2

    fig = plt.figure()
    ax = fig.add_subplot(121)

    while current_seqlen <= 10: # end until current_seqlen = 30
        start = (current_seqlen-2)*40
        end = (current_seqlen-1)*40
        label_ = 'Length (' + str(current_seqlen) + ')'

        ax.plot(wsize[start:end], unordered_fscore[start:end], color='r', linewidth=2, label=label_)
        # windows are not sorted. So, getting bad result.
        current_seqlen += 1

    plt.legend()
    plt.grid(True)
    plt.show()

