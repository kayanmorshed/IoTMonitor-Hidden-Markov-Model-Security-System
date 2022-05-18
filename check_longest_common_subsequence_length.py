import pandas as pd
import numpy as np


def LCS(dataset,decoded_sequence):
    # get event sequence from the dataset and then convert into list
    event_seq = list(dataset['event_value_numerical'].values)  # get the list of the events in the dataset

    # convert the event_seq to string X
    event_string = ''
    for item in event_seq:
        event_string += str(item)

    # convert the decoded_sequence list from numpy.float64 into numpy.int64 and convert into list
    dcd_seq_int = list(decoded_sequence.astype(np.int64))

    # convert the dcd_seq to string
    dcd_seq_string = ''
    for item in dcd_seq_int:
        dcd_seq_string += str(item)

    event_string_length = len(event_string) # length of the event_string (m)
    dcd_seq_string_length = len(dcd_seq_string) # length of the dcd_seq_string (n)

    counter = [[0]*(dcd_seq_string_length+1) for x in range(event_string_length+1)]

    longest = 0
    lcs_str = set()

    for i in range(event_string_length):
        for j in range(dcd_seq_string_length):
            if event_string[i] == dcd_seq_string[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_str = set()
                    longest = c
                    lcs_str.add(event_string[i-c+1:i+1])
                elif c == longest:
                    lcs_str.add(event_string[i-c+1:i+1])

    # return the length of the event sequence string and length of the longest common subsequence string
    return len(event_seq), len(lcs_str.pop())
