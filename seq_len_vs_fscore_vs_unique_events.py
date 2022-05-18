import pandas as pd
import numpy as np
import os


# define the method seqlen_vs_fscore()
def seqlen_vs_fscore (csv_dir, dst_name):
    # read the dataset with the name "dst_name" from the directory "csv_dir"
    input_dataset = pd.read_csv(os.path.join(csv_dir, dst_name))

    # extract the column "event_value_numerical" from input_dataset
    event_value_numerical = input_dataset['event_value_numerical']

    # get the length of the column "event_value_numerical"
    lenth_of_event_sequence = len(event_value_numerical)

    # get the unique numerical values (i.e. unique events) present in the "event_value_numerical" column
    # and then determine the length of the unique values array
    number_of_unique_events = len(event_value_numerical.unique())

    return lenth_of_event_sequence, number_of_unique_events


if __name__ == '__main__':
    # get the directory "dataset"
    dataset_directory = os.path.join(os.getcwd(), 'window_1', 'temp')

    # read the dataset to write and get the list of the dataset names
    output_dst = pd.read_csv("window_1/estimating_vs_decoding_time_w1.csv")
    dataset_name_list = output_dst['dataset_name']
    fscore = output_dst['f-score'] # get the list of the fscores

    # get all the file names with the extension .csv and put into a list
    file_name_list = []
    for file_name in os.listdir(dataset_directory):
        if '.csv' in file_name:
            file_name_list.append(file_name)

    # create an empty dataframe and set the idx = 0
    dframe = pd.DataFrame(data={'dataset_name': [], 'seq_len': [], 'num_unique_events': [], 'f-score': []})
    idx = 0

    # iterate the method "seqlen_vs_fscore()" for all the csvs containing name in "file_name_list"
    for i in range(0, len(file_name_list)):
        seq_len, num_unique_events = seqlen_vs_fscore(dataset_directory, file_name_list[i])

        # get the index of the file_name (dataset_name) from the "estimating_vs_decoding_time_w*.csv" dataset
        index_output_dst = list(dataset_name_list).index(file_name_list[i])

        # get the corresponding f-score from the "estimating_vs_decoding_time_w*.csv" dataset using the index
        corresponding_fscore = fscore[index_output_dst]

        # write to the dataframe "dframe"
        dframe.loc[idx] = [file_name_list[i], seq_len, num_unique_events, corresponding_fscore]
        idx += 1 # increase the index for the dataframe

    # convert the dataframe into a csv file
    dframe.to_csv('window_1/seqlen_vs_score.csv', index=False)

    # read the just created dataset
    read_dst = pd.read_csv('window_1/seqlen_vs_score.csv')

    # sort the dataset in ascending order based on sequence length
    dst_sorted_seqlen = read_dst.sort_values("seq_len")
    dst_sorted_seqlen.to_csv('window_1/seqlen_vs_score_sorted_by_seqlen.csv', index=False)

    # sort the dataset in ascending order based on number of unique events
    dst_sorted_num_unique_events = read_dst.sort_values("num_unique_events")
    dst_sorted_num_unique_events.to_csv('window_1/seqlen_vs_score_sorted_by_num_unique_events.csv', index=False)



