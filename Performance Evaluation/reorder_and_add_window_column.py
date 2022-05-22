import pandas as pd
import numpy as np
import os
import re


if __name__ == '__main__':

    # define the length size
    length = 2

    # set the directory where folders of the csv files are located
    # wdir = os.path.join(os.getcwd(), 'length-based', 'v2')
    wdir = os.path.join(os.getcwd(), 'c3')
    # wdir = os.path.join(os.getcwd(), 'c3')

    while length <= 30:
        # read tge performance_eval dataset
        input_eval_dst = pd.read_csv(os.path.join(wdir, 'l' + str(length) + '_eval_exec_no_1.csv'))
        # input_eval_dst = pd.read_csv(os.path.join(wdir, 'performance_eval_l' + str(length) + '.csv'))

        # read the dataset names existed in a window
        entry_dst_name= input_eval_dst['dataset_name']

        # empty list to store retrieve window sizes from each row
        window_list = []

        # populate the length list with zero
        for i in range(len(entry_dst_name)):
            # get the window size from the dataset name
            # search_wsize = re.search('window_(.*)_length', entry_dst_name[i])
            # search_wsize = re.search('w_(.*)_l', entry_dst_name[i])
            # search_wsize = re.search('window_(.*)_no', entry_dst_name[i]) # for case C2, C4
            search_wsize = re.search('w_(.*)_l', entry_dst_name[i]) # for case C1. C3
            wsize = int(search_wsize.group(1))  # window size in String -> int
            window_list.append(wsize)

        # create a dataframe
        dframe = pd.DataFrame(data={'window_size': window_list})

        # concatenate dframe with input_eval_dst
        df = pd.concat([input_eval_dst, dframe], axis=1)

        # sort the newly create dataframe
        df_sorted = df.sort_values('window_size')

        # print(df_sorted.head())

        # specify the path to check whether there is such directory named 'sorted'
        path_to_check = os.path.join(wdir, 'sorted_combined')

        # create the directory 'sorted' if it does not exist
        if not os.path.exists(path_to_check):
            # create a directory on this path
            os.makedirs(path_to_check)

        # convert the sorted dataframe into a csv
        df_sorted.to_csv(os.path.join(path_to_check, 'l' + str(length) + '_eval_exec_sorted.csv'), index=False)

        # increase the length by 1
        length += 1



