import pandas as pd
import numpy as np
import os


if __name__ == '__main__':
    # set the csv directory
    csv_dir = os.path.join(os.getcwd(), 'c3')

    # get the filenames of all csvs
    filenames = [fname for fname in os.listdir(os.path.join(csv_dir, 'sorted')) if '.csv' in fname]

    # sefine some emptyb lists to contain necessary column values from the dataset
    col_dset = []
    col_wsize = []
    col_event_seqlen = []
    col_fscore = []
    col_unordered_fscore = []

    # create an empty dataframe to hold the columns 'fscore', 'unordered_fscore', 'event_seq_len', and 'window_size'
    # dframe = pd.DataFrame(data={'dataset_name': [], 'window_size': [], 'event_seq_len': [], 'f-score': [], 'unordered_fscore': []})

    # read each dataset and stack the corresponding columns into dframe
    for fitem in filenames:
        # read the datset
        dst = pd.read_csv(os.path.join(csv_dir, 'sorted', fitem))

        for i in range(0, dst.shape[0]):
            # get the columns we are interested in
            col_dset.append((dst['dataset_name'].values)[i])
            col_wsize.append((dst['window_size'].values)[i])
            col_event_seqlen.append((dst['event_seq_len'].values)[i])
            col_fscore.append((dst['f-score'].values)[i])
            col_unordered_fscore.append((dst['unordered_fscore'].values)[i])

    # # populate 'dframe'
    # for i in range(len(col_dset)):
    #     dframe.loc[i] = [col_dset[i], col_wsize[i], col_event_seqlen[i], col_fscore[i], col_unordered_fscore[i]]

    # create an empty dataframe to hold the columns 'fscore', 'unordered_fscore', 'event_seq_len', and 'window_size'
    dframe = pd.DataFrame(data={'dataset_name': col_dset, 'window_size': col_wsize, 'event_seq_len': col_event_seqlen, 'f-score': col_fscore, 'unordered_fscore': col_unordered_fscore})

    # sort the newly create dataframe based on 'sequence length'
    df_sorted = dframe.sort_values('event_seq_len')

    # check whether there is a directory named "fscore"
    path_fscore = os.path.join(csv_dir, 'fscore')

    if not os.path.exists(path_fscore):
        os.makedirs(path_fscore, )

    # convert 'dframe' into a csv
    df_sorted.to_csv(os.path.join(path_fscore, 'fscore_wsize_seqlen_c3.csv'), index=False)





