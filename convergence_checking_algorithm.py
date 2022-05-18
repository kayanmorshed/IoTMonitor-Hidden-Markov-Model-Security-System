import numpy as np


# ***********************************************************************************************************
# ***************************** Determining whether convergence occurs ****************************************
# ***********************************************************************************************************
def convergence_checker(trans_before_iteration, emission_before_iteration, trans_after_iteration, emission_after_iteration, conv_thresh):
    trans_flag = 0 # trans_flag = 0 means there is NO element in transition probability matrix
                   # whose value change after iteration greater than the threshold
    emis_flag = 0 # emis = 0 means there is NO element in emission probability matrix
                  # whose value change after iteration greater than the threshold

    checker_status = 0 # zero means convergence did not happen

    trans_diff = abs(np.subtract(trans_after_iteration, trans_before_iteration))
    emis_diff = abs(np.subtract(emission_after_iteration, emission_before_iteration))

    # print("\n trans diff: ", trans_diff)
    # print("\n emis diff: ", emis_diff)

    # for i in range(0, trans_before_iteration.shape[0]):
    #     for j in range(0, trans_before_iteration.shape[1]):
    #         if (trans_after_iteration[i, j] - trans_before_iteration[i, j]) > conv_thresh: # check whether there is any element greater than the allowed threshold
    #             trans_flag = 1
    #             break
    #
    # for i in range(0, emis_before_iteration.shape[0]):
    #     for j in range(0, emis_before_iteration.shape[1]):
    #         if (emis_after_iteration[i, j] - emis_before_iteration[i, j]) > conv_thresh: # check whether there is any element greater than the allowed threshold
    #             emis_flag = 1
    #             break

    for i in range(0, trans_diff.shape[0]):
        for j in range(0, trans_diff.shape[1]):
            if trans_diff[i, j] > conv_thresh: # check whether there is any element greater than the allowed threshold
                trans_flag = 1
                break
            else:
                pass

    for i in range(0, emis_diff.shape[0]):
        for j in range(0, emis_diff.shape[1]):
            if emis_diff[i, j] > conv_thresh: # check whether there is any element greater than the allowed threshold
                emis_flag = 1
                break
            else:
                pass

    if trans_flag == 0 and emis_flag == 0:
        checker_status = 1 # checker_status=1 means convergence occurs
    else:
        pass

    return checker_status
