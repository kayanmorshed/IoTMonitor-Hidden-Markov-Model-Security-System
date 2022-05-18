import numpy as np
from log_matrix_operations import get_actual_values_from_log2_2d


# *********************************************************************************
# ******************************* Viterbi Algorithm *******************************
# *********************************************************************************
def viterbi(obseq, trans, emis, init_dist, event_list):
    dataset_rows = obseq.shape[0]
    state_transition_rows = trans.shape[0]

    omega = np.zeros((dataset_rows, state_transition_rows))
    # omega[0, :] = np.log(init_dist * emis[:, obseq[0]]) # original
    # modified, but unchecked version
    for i in range(state_transition_rows):
        # omega[0,i] = np.log2(init_dist[i]) + np.log2(emis[i, obseq[0]])
        omega[0,i] = init_dist[i] + emis[i, obseq[0]]

    prev = np.zeros((dataset_rows - 1, state_transition_rows))

    for t in range(1, dataset_rows):
        for j in range(state_transition_rows):
            # Same as Forward Probability
            # probability = omega[t - 1] + np.log2(trans[:, j]) + np.log2(emis[j, obseq[t]])
            probability = omega[t - 1] + trans[:, j] + emis[j, obseq[t]]

            # This is our most probable state given previous state at time t (1)
            prev[t - 1, j] = np.argmax(probability)

            # This is the probability of the most probable state (2)
            omega[t, j] = np.max(probability)

    # Path Array
    path_array = np.zeros(dataset_rows)

    # Find the most probable last hidden state
    last_state = np.argmax(omega[dataset_rows - 1, :])

    path_array[0] = last_state

    backtrack_index = 1
    for i in range(dataset_rows - 2, -1, -1):
        path_array[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1

    # *************** To test the index of path_array elements ********
    # print("\n Backtrack Sequence (before flipping): ", path_array)
    #
    # for i in range(0, len(path_array)):
    #     print("\n Index of %d in the original dataset is: " % path_array[i], np.where(obseq == path_array[i]))
    # *************** Test completes here *****************************

    # Flip the path array since we were backtracking
    path_array = np.flip(path_array, axis=0)
    # print("\n Decoded Observation Sequence: \n", path_array) # provides observation sequence, not event sequence
    # *******************************************************************************************
    # ************************* Have to map observation sequence with event seq *****************
    # *******************************************************************************************
    return path_array


    # # Convert numeric values to actual hidden states
    # result = []
    #
    # # for x in path_array:
    # #     # if x == 0:
    # #     #     result.append("A")
    # #     # else:
    # #     #     result.append("B")
    #
    # for y in range(0, len(path_array)):
    #     # this gives only the sequence of events as in original datatset, not based on the sequence against observation sequence
    #     result.append(event_list[y])
    #
    # return result