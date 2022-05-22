import os
import pandas as  pd
import numpy as np

if __name__ == '__main__':
    # get the dataset directory
    wdir = os.path.join(os.getcwd(), 'c3')

    # read the dataset
    dst = pd.read_csv(os.path.join(wdir, 'all_lengths_timestep_conv_min.csv'))

    # get the columns
    num_states = dst['num_states']
    num_observ = dst['num_observ']
    timestep_conv = dst['timestep_conv']

    # take the ratio between num_observations and num_states
    states_obsrv_ratio = num_observ/num_states

    # create a dataframe
    dframe = pd.DataFrame(data={'states_obsrv_ratio': states_obsrv_ratio})

    # concatenate dframe with 'dst'
    df = pd.concat([dst, dframe], axis=1)

    # sort the newly create dataframe
    # df_sorted = df.sort_values('states_obsrv_ratio')

    # convert the newly formed dataframe into a csv file
    df.to_csv(os.path.join(wdir, 'all_lengths_timestep_min_ratio.csv'))
