import os
import pandas as  pd
import numpy as np

if __name__ == '__main__':
    # get the dataset directory
    wdir = os.path.join(os.getcwd(), 'c3')

    # read the dataset
    dst = pd.read_csv(os.path.join(wdir, 'all_lengths_timestep_min_ratio.csv'))

    # get the columns
    states_obsrv_ratio = dst['states_obsrv_ratio']
    timestep_conv = dst['timestep_conv']

    # declare a dictionary to hold all timestep values for the same ratio
    ratio_dict = {}

    # populate the dictionary
    for i in range(0, states_obsrv_ratio.shape[0]):
        if states_obsrv_ratio[i] not in ratio_dict.keys(): # no "states_obsrv_ratio" in the keys yet
            ratio_dict[states_obsrv_ratio[i]] = [timestep_conv[i]]
        else:
            ratio_dict[states_obsrv_ratio[i]].append(timestep_conv[i])

    # ************ Take the minimum timestep value for a certain ratio ******************
    ratio_min = [] # to hold the keys of ratio_dict
    timestep_min = [] # to hold the minimum value from the value list against a key

    for key in ratio_dict.keys():
        ratio_min.append(key)
        timestep_min.append(min(ratio_dict[key]))

    # create a dataframe
    dframe_min = pd.DataFrame(data={'states_obsrv_ratio': ratio_min, 'timestep_conv': timestep_min})

    # sort the dframe_min based on state_obsrv_raito
    dframe_min_sorted = dframe_min.sort_values('states_obsrv_ratio')

    # convert the sorted dataframe into a CSV file
    dframe_min_sorted.to_csv(os.path.join(wdir, 'timestep_min_ratio_no_duplicates.csv'))

    # ************ Take the maximum timestep value for a certain ratio ******************
    ratio_max = [] # to hold the keys of ratio_dict
    timestep_max = [] # to hold the minimum value from the value list against a key

    for key in ratio_dict.keys():
        ratio_max.append(key)
        timestep_max.append(max(ratio_dict[key]))

    # create a dataframe
    dframe_max = pd.DataFrame(data={'states_obsrv_ratio': ratio_max, 'timestep_conv': timestep_max})

    # sort the dframe_max based on state_obsrv_raito
    dframe_max_sorted = dframe_max.sort_values('states_obsrv_ratio')

    # convert the sorted dataframe into a CSV file
    dframe_max_sorted.to_csv(os.path.join(wdir, 'timestep_max_ratio_no_duplicates.csv'))



