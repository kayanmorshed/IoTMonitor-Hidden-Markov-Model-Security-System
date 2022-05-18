import numpy as np
import pandas as pd
import math
import os
import timeit
import matplotlib.pyplot as plt
from parameter_estimation_algorithm import baum_welch
from decoding_algorithm import viterbi
from check_false_positives_and_negatives import check_precision_recall_fscore, unordered_accuracy_score
from check_longest_common_subsequence_length import LCS

if __name__ == '__main__':

    # window_list = [110, 120, 130, 140, 150]
    length_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    # length_list = [0]

    # for item in window_list:
    for item in length_list:
        # define execution number
        exec_attempt = 1
        while exec_attempt <= 10:
            # set the path to dataset
            path_to_dataset = os.path.join(os.getcwd(), 'workspace', '2019-04-04', 'updated_july23', 'c3', 'l_' + str(item))

            # open the .txt file to write reports
            file_to_write = open(os.path.join(path_to_dataset, 'exec_no_' + str(exec_attempt) + '.txt'), "a")

            # define a DataFrame to contain execution-time and decoding_time data
            # dframe = pd.DataFrame(data={'dataset_name': [], 'estimating_time': [], 'decoding_time': []})
            dframe = pd.DataFrame(data={'dataset_name': [], 'estimating_time': [], 'decoding_time': [], 'precision': [], 'recall': [], 'f-score': [], 'unordered_fscore': [], 'event_seq_len': [], 'lcs_len': []})
            dframe_index = 0

            # **************************************** Read datasets from a directory *********************/
            # **********************************************************************************************
            all_csv = [file_name for file_name in os.listdir(path_to_dataset) if 'numerical.csv' in file_name]

            # append dataframe to the list
            for filename in all_csv:
                data = pd.read_csv(os.path.join(path_to_dataset, filename))
                print("\n Dataset Name: ", filename)

                # write the name of the dataset in the "hmm_take1.txt" file
                file_to_write.write("\n Dataset Name: %s" % filename)

                event_sequence = data['event_value_numerical'].values  # get the list of the events in the dataset
                observation_sequence = data['sensor_value_numerical'].values  # get the observation sequence (0, 1, 2,.........)
                # print("observation sequence unique: ", data.sensor_value_numerical.unique())

                # set the number of states N and number of observations M
                number_of_states = len(
                    data.event_value_numerical.unique())  # unique events for extracted_dataset_1st_hour_window_5.csv
                number_of_observations = len(
                    data.sensor_value_numerical.unique())  # unique sensors for extracted_dataset_1st_hour_window_5.csv
                # print("number of states: ", number_of_states)
                # print("number of observations: ", number_of_observations)

                # generating transition probability matrix using normal distribution
                # mu, sigma = 0.5, 0.05  # mean and standard deviation
                mu, sigma = 0.5, 0.05  # mean and standard deviation
                np.random.seed(0)  # seed = 0 ensures the same random numbers for every execution
                state_transition_probabilities = np.random.normal(mu, sigma, size=(number_of_states, number_of_states))
                # normalize the probabilities row-wise so that sum(probabilities in a row) = 1
                state_transition_probabilities /= np.sum(state_transition_probabilities, axis=1)[:, np.newaxis]

                # # generating transition probability matrix using dirichlet distribution
                # alpha = np.random.random(number_of_states) # cols in generated matrix
                # size = number_of_states # rows in generated matrix
                # state_transition_probabilities = np.random.dirichlet(alpha, size)

                # # generating emission probability matrix using normal distribution
                # mu, sigma = 0.5, 0.05  # mean and standard deviation
                # np.random.seed(50)  # seed = 0 ensures the same random numbers for every execution
                # emission_probabilities = np.random.normal(mu, sigma, size=(number_of_states, number_of_observations))
                # # normalize the probabilities row-wise so that sum(probabilities in a row) = 1
                # emission_probabilities /= np.sum(emission_probabilities, axis=1)[:, np.newaxis]

                # generating emission probability matrix using dirichlet distribution
                alpha = np.random.random(number_of_observations)  # cols in generated matrix
                size = number_of_states  # rows in generated matrix
                emission_probabilities = np.random.dirichlet(alpha, size)


                # generating initial distribution matrix using normal distribution
                # mu, sigma = 0.5, 0.05 # mean and standard deviation
                mu, sigma = 0.5, 0.05 # mean and standard deviation
                np.random.seed(0)  # seed = 0 ensures the same random numbers for every execution
                initial_distribution = np.random.normal(mu, sigma, number_of_states)
                # normalize the probabilities row-wise so that sum(probabilities in a row) = 1
                initial_distribution /= np.sum(initial_distribution)

                # # generating emission probability matrix using dirichlet distribution
                # alpha = np.random.random(number_of_states)  # cols in generated matrix
                # # size = 1  # rows in generated matrix
                # initial_distribution = np.random.dirichlet(alpha)

                # set the number of iterations over which baum-welch will execute
                number_of_iteration = 1000
                file_to_write.write("\n\n Number of iteration: %d" % number_of_iteration)

                # set the convergence threshold value
                convergence_threshold = 0.000001
                # convergence_threshold = 0.0000000001
                file_to_write.write("\n\n Convergence Threshold: %0.20f" % convergence_threshold)

                # print("Selected convergence threshold: ", convergence_threshold)

                # print("****************************************************")

                # Call "baum_welch" function from the "parameter_estimation_algorithm" module
                start_time_baum_welch = timeit.default_timer()

                # trans_updated, emis_updated, timestep_converge = baum_welch(observation_sequence,
                #                                                             state_transition_probabilities,
                #                                                             emission_probabilities, initial_distribution,
                #                                                             number_of_iteration, convergence_threshold)

                trans_updated, emis_updated, init_dist_updated, timestep_converge = baum_welch(observation_sequence, state_transition_probabilities, emission_probabilities, initial_distribution, number_of_iteration, convergence_threshold)

                # write the "timestep to converge" to the file "hmm_take1.txt"
                file_to_write.write("\n\n Converged at timestemp: %d" % timestep_converge)

                estimating_time = timeit.default_timer() - start_time_baum_welch

                # write the estimating time to the file "hmm_take1.txt"
                file_to_write.write("\n\n Probability estimation (BW) time: %0.6f" % estimating_time)

                # set off the print_options for the numpy array elements
                np.set_printoptions(formatter=None)

                # Call "viterbi" algorithm from the "decoding_algorithm" module
                start_time_viterbi = timeit.default_timer()
                # has to conver viterbi() logs into log2
                # decoded_event_sequence = viterbi(observation_sequence, state_transition_probabilities,
                #                                  emission_probabilities, initial_distribution, event_sequence)

                decoded_event_sequence = viterbi(observation_sequence, trans_updated, emis_updated, np.log2(initial_distribution), event_sequence)

                # write the observation sequence to the "hmm_take_1.txt"
                file_to_write.write("\n\n Observation sequence: ")
                for i in observation_sequence:
                    file_to_write.write("%s," % i.astype(np.int64))

                # write the event sequence to the "hmm_take_1.txt"
                file_to_write.write("\n\n Original event sequence: ")
                for i in event_sequence:
                    file_to_write.write("%s," % i.astype(np.int64))

                # write the decoded event sequence to the "hmm_take_1.txt"
                file_to_write.write("\n\n Decoded event sequence: ")
                for i in decoded_event_sequence:
                    file_to_write.write("%s," % i.astype(np.int64))

                decoding_time = timeit.default_timer() - start_time_viterbi
                # print("\n Execution time to extract the hidden sequence: ", decoding_time)

                # write the decoding time to the file "hmm_take1.txt"
                file_to_write.write("\n\n Decoding (Viterbi) time: %0.6f" % decoding_time)

                # check false positives
                precision, recall, fscore = check_precision_recall_fscore(data, decoded_event_sequence)

                # check unrdered f-score
                unordered_fscore = unordered_accuracy_score(data, decoded_event_sequence)

                # get the longest common subsequence length
                event_sequence_length, lcs_length = LCS(data, decoded_event_sequence)

                # write the estimating time and decoding time data against the dataset name in a DataFrame
                dframe.loc[dframe_index] = [filename, estimating_time, decoding_time, precision, recall, fscore, unordered_fscore, event_sequence_length, lcs_length]
                dframe_index += 1  # increase the index after containg data for each item

                file_to_write.write("\n\n Length of sequence: %0.2f" %len(decoded_event_sequence))
                file_to_write.write("\n\n Precision: %0.3f" %precision)
                file_to_write.write("\n\n Recall: %0.3f" %recall)
                file_to_write.write("\n\n F-score: %0.3f" %fscore)
                file_to_write.write("\n\n ************************************************************* \n\n")

            # close the file
            file_to_write.close()

            # convert the DataFrame into a CSV file
            # dframe.to_csv(path_to_dataset + 'estimating_vs_decoding_time_w' + str(item) + '.csv', index=False)
            dframe.to_csv(os.path.join(path_to_dataset, 'eval_exec_no_' + str(exec_attempt) + '.csv'), index=False)

            # reset the dataframe
            # dframe = dframe[0:0]

            # increment the execution run
            exec_attempt += 1

