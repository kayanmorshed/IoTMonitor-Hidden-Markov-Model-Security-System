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
    wdir_c3 = os.path.join(os.getcwd(), 'c3') # for c3: no undetectables, but race conditions

    # draw the figure
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_xlabel('Ratio between number of unique observation states \n and number of unique true states', fontsize=11)
    ax.set_ylabel('Number of iterations', fontsize=11)
    # ax.set_title('Number of unique observation states')

    # ax2 = fig.add_subplot(122)
    # ax2.set_xlabel('Number of unique observation states', fontsize=12)
    # ax2.set_ylabel('Timestep to estimate comverged proabilities', fontsize=12)
    # # ax.set_title('Number of unique observation states')

    # read the eval dataset
    # dst = pd.read_csv(os.path.join(wdir_c3, 'all_lengths_timestep_conv_min.csv'))
    # dst = pd.read_csv(os.path.join(wdir_c3, 'all_lengths_timestep_min_ratio.csv')) # for ratio
    # dst = pd.read_csv(os.path.join(wdir_c3, 'timestep_min_ratio_no_duplicates.csv')) # for ratio with no duplicates but min
    # dst = pd.read_csv(os.path.join(wdir_c3, 'timestep_max_ratio_no_duplicates.csv')) # for ratio with no duplicates but min
    dst = pd.read_csv(os.path.join(wdir_c3, 'timestep_ratio_no_duplicates_manually_customized.csv')) # for ratio with no duplicates but min

    # get the columns
    # num_states = dst['num_states']
    # num_observ = dst['num_observ']
    timestep_conv = dst['timestep_conv']
    states_obsrv_ratio = dst['states_obsrv_ratio']

    # interpolation: wsize
    # wsize_new = np.linspace(wsize.min(), wsize.max(), 40)

    # interpolate estimating time with window_new
    # est_temp = make_interp_spline(wsize, est, k=3)

    # choose hex for the randomly chosen color
    color = "%06x" % random.randint(0, 0xFF01EF)
    color2 = "%06x" % random.randint(0, 0xFF04FF)

    # label_ = 'length = ' + str(np.int64(seqlen[0])) # label for each line
    # label_box.append('l-' + str(np.int64(seqlen[0]))) # for box plots

    # ax.scatter(num_states, timestep_conv, marker='o', color='red') # original
    # ax.scatter(states_obsrv_ratio, timestep_conv, marker='o', color='red') # for states_observation_ratio
    ax.plot(states_obsrv_ratio, timestep_conv, color='red', marker='o', markersize=4, linewidth=2.0) # for states_observation_ratio
    ax.legend()
    # Show the major grid lines with dark grey lines
    ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    # Show the minor grid lines with very faint and almost transparent grey lines
    ax.minorticks_on()
    ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
    # ax.show()

    # # ax.plot(num_states, timestep_conv, 'ro', linewidth=2) # without interpolation
    # ax2.scatter(num_observ, timestep_conv, marker='o', color='red')  # without interpolation
    # ax2.legend()
    # # Show the major grid lines with dark grey lines
    # ax2.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    # # Show the minor grid lines with very faint and almost transparent grey lines
    # ax2.minorticks_on()
    # ax2.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
    # # ax.show()

    plt.tight_layout(h_pad=0.1)

    plt.show()
