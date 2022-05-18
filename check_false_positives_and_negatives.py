import pandas as pd
import numpy as np
import os
from collections import Counter
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix


def check_precision_recall_fscore(dataset, decoded_sequence):
    # get event sequence from the dataset
    # event_seq = dataset['event_value_numerical'].values  # get the list of the events in the dataset
    event_seq = dataset['event_value_numerical'].values  # hence, decoded_seq actually is the obsrv_seq
    observation_seq = dataset['sensor_value_numerical'].values  # get the observation sequence (0, 1, 2,.........)

    # # create a hashmap between event_seq and observation_seq
    # event_sensor_map = {}
    # for i in range(len(event_seq)):
    #     if event_seq[i] in event_sensor_map.keys():
    #         if observation_seq[i] not in event_sensor_map[event_seq[i]]:
    #             event_sensor_map[event_seq[i]].append(observation_seq[i])
    #     else:
    #         event_sensor_map[event_seq[i]] = [observation_seq[i]]

    # convert the decoded_sequence list from numpy.float64 into numpy.int64
    dcd_seq_int = decoded_sequence.astype(np.int64)

    # # extract the corresponding event sequence using event_sensor_map from decoded_sequence
    # extracted_event_seq = np.zeros(len(dcd_seq_int))
    # for i in range(len(dcd_seq_int)):
    #     for item in event_sensor_map.keys(): # which key in the dictionary has values equal to dec_seq_int
    #         if dcd_seq_int[i] in event_sensor_map[item]:
    #             extracted_event_seq[i] = item
    #             break

    # print("\n Extracted Event Sequence: \n", extracted_event_seq)
    # print("\n Extracted Event Sequence Length: \n", len(extracted_event_seq))
    # print("\n Event Sensor Map: \n", event_sensor_map)

    # calculate prediction
    precision = precision_score(event_seq, dcd_seq_int, average='weighted')
    # precision = precision_score(observation_seq, dcd_seq_int, average='weighted') # using observation sequence; may or may not be the actual case
    # precision = precision_score(event_seq, extracted_event_seq, average='micro')
    # print('Precision: %.3f' % precision)

    # calculate recall
    recall = recall_score(event_seq, dcd_seq_int,  average='weighted')
    # recall = recall_score(observation_seq, dcd_seq_int,  average='weighted')
    # recall = recall_score(event_seq, extracted_event_seq,  average='micro')
    # print('Recall: %.3f' % recall)

    # calculate score
    score = f1_score(event_seq, dcd_seq_int, average='micro')
    # score = f1_score(observation_seq, dcd_seq_int, average='micro')
    # score = f1_score(event_seq, extracted_event_seq, average='micro')
    # print('F-Measure: %.3f' % score)

    # print("\n Average precison score: %.3f" % average_precision_score(extracted_event_seq, dcd_seq_int))
    # print("\n Classification report: \n", classification_report(extracted_event_seq, dcd_seq_int, target_names= [i for i in event_sensor_map.keys()]))

    return precision, recall, score


def unordered_accuracy_score (dataset, decoded_sequence):
    # get event sequence from the dataset
    event_seq = dataset['event_value_numerical'].values  # get the list of the events in the dataset
    observation_seq = dataset['sensor_value_numerical'].values  # get the observation sequence (0, 1, 2,.........)

    # convert the decoded_sequence list from numpy.float64 into numpy.int64
    dcd_seq_int = decoded_sequence.astype(np.int64)

    # create Counter for both the arrays
    event_counter = Counter(event_seq)
    dcd_seq_counter = Counter(dcd_seq_int)

    # get the counter of the missing events in the decoded sequence
    diff_in_events = event_counter - dcd_seq_counter

    # count the total events in the actual event sequence and also in the missing sequence
    total_events = sum(event_counter.values())
    event_not_found = sum(diff_in_events.values())

    # print(event_counter)
    # print(dcd_seq_counter)
    # print(diff_in_events)
    # print(total_events)
    # print(event_not_found)
    unordered_score = 1 - (event_not_found/total_events)

    # return (1 - (event_not_found/total_events))
    return unordered_score



def check_fp_fn_counter(dataset, decoded_sequence):
    # get event sequence from the dataset
    event_seq = dataset['event_value_numerical'].values  # get the list of the events in the dataset
    # observation_seq = dataset['sensor_value_numerical'].values  # get the observation sequence (0, 1, 2,.........)

    # convert the decoded_sequence list from numpy.float64 into numpy.int64
    dcd_seq_int = decoded_sequence.astype(np.int64)

    # create Counter for both the arrays
    event_counter = Counter(event_seq)
    dcd_seq_counter = Counter(dcd_seq_int)

    false_negative = 0
    false_positive = 0

    for item in event_counter:
        if (event_counter[item] - dcd_seq_counter[item]) < 0:
            false_positive += abs(event_counter[item] - dcd_seq_counter[item])
        else:
            false_negative += event_counter[item] - dcd_seq_counter[item]

    # print("Event counter: ", event_counter)
    # print("Decoded seq counter: ", dcd_seq_counter)

    return false_positive, false_negative


# def check_false_positives_in_exact_sequence(dataset, decoded_sequence):
#     # get event sequence from the dataset
#     event_seq = dataset['event_value_numerical'].values  # get the list of the events in the dataset
#     # observation_seq = dataset['sensor_value_numerical'].values  # get the observation sequence (0, 1, 2,.........)
#
#     nan_value = event_seq[1] # 2nd element of "event_seq" is always "nan"
#
#     idx = 0 # to track the index of ndarray
#     count = 0 # to count all the true positive values
#     total = 0 # to count all the non-nan values
#
#     while idx < len(event_seq):
#         if event_seq[idx] != nan_value:
#             if event_seq[idx] == max(event_seq):
#                 print(idx)
#             if event_seq[idx] == decoded_sequence[idx]:
#                 count += 1
#             total += 1
#             idx += 1
#         else:
#             idx += 1
#
#     # calculate fp_rate
#     fp_rate = 100 - (count/total) * 100
#
#     return fp_rate
