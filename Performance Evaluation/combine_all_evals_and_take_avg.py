import pandas as pd
import numpy as np
import os
import re


if __name__ == '__main__':

    # define the length size
    length = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

    for item in length:
        # set the directory where folders of the csv files are located
        wdir = os.path.join(os.getcwd(), 'c3', 'l_' + str(item))

        # create a dataframe to hold the avg of all columns across all datasets
        # dframe = pd.DataFrame(data={'dataset_name':[], 'estimating_time': [], 'decoding_time': [],
        #                             'precision': [], 'recall': [], 'f-score': [], 'unordered_fscore': [],
        #                             'event_seq_len': [], 'lcs_len': []})

        dset_name = np.zeros(40)
        est_time = np.zeros(40, dtype=np.float64)
        dcd_time = np.zeros(40, dtype=np.float64)
        precision = np.zeros(40, dtype=np.float64)
        recall = np.zeros(40, dtype=np.float64)
        fscore = np.zeros(40, dtype=np.float64)
        unordered_fscore = np.zeros(40, dtype=np.float64)
        # event_seqlen = np.zeros(40, dtype=np.float64)
        # lcs_len = np.zeros(40, dtype=np.float64)

        # get the name of the eval datasets
        filenames = [fname for fname in os.listdir(wdir) if 'eval_exec_no' in fname]

        for filename in filenames:
            # set the index of the created dataframe
            # dframe_index = 0

            # read the performance_eval dataset
            input_eval_dst = pd.read_csv(os.path.join(wdir, filename))

            dset_name = input_eval_dst['dataset_name']
            est_time += input_eval_dst['estimating_time']
            dcd_time += input_eval_dst['decoding_time']
            precision += input_eval_dst['precision']
            recall += input_eval_dst['recall']
            fscore += input_eval_dst['f-score']
            unordered_fscore += input_eval_dst['unordered_fscore']
            # event_seqlen += input_eval_dst['event_seq_len']
            # lcs_len += input_eval_dst['lcs_len']

        # create a dataframe to hold the avg of all columns across all datasets
        dframe = pd.DataFrame(data={'dataset_name': dset_name, 'estimating_time': est_time/10, 'decoding_time': dcd_time/10, 'precision': precision/10, 'recall': recall/10, 'f-score': fscore/10, 'unordered_fscore': unordered_fscore/10})

        dframe.to_csv(os.path.join(wdir, 'eval_exec_combined_l' + str(item) +'.csv'), index=False)


            # # empty list to store retrieve window sizes from each row
            # window_list = []
            #
            # # populate the length list with zero
            # for i in range(len(entry_dst_name)):
            #     # get the window size from the dataset name
            #     # search_wsize = re.search('window_(.*)_length', entry_dst_name[i])
            #     # search_wsize = re.search('w_(.*)_l', entry_dst_name[i])
            #     # search_wsize = re.search('window_(.*)_no', entry_dst_name[i]) # for case C2, C4
            #     search_wsize = re.search('w_(.*)_l', entry_dst_name[i]) # for case C1. C3
            #     wsize = int(search_wsize.group(1))  # window size in String -> int
            #     window_list.append(wsize)
            #
            # # create a dataframe
            # dframe = pd.DataFrame(data={'window_size': window_list})
            #
            # # concatenate dframe with input_eval_dst
            # df = pd.concat([input_eval_dst, dframe], axis=1)

            # # sort the newly create dataframe
            # df_sorted = df.sort_values('window_size')
            #
            # # print(df_sorted.head())
            #
            # # specify the path to check whether there is such directory named 'sorted'
            # path_to_check = os.path.join(wdir, 'sorted')
            #
            # # create the directory 'sorted' if it does not exist
            # if not os.path.exists(path_to_check):
            #     # create a directory on this path
            #     os.makedirs(path_to_check)
            #
            # # convert the sorted dataframe into a csv
            # df_sorted.to_csv(os.path.join(path_to_check, 'performance_eval_l' + str(length) + '_sorted.csv'), index=False)
            #
            # # increase the length by 1
            # length += 1



