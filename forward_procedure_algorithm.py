import numpy as np
from log_matrix_operations import log_space_product


# ***************************************************************************************************************
# ****************** define the function forward procedure to calculate the probability of **********************
# ***************** being at a certain state given all the previous observation up until this state *************
# ***************************************************************************************************************
def forward_procedure(obseq, trans_log2, emis_log2, init_dist_log2):
    alpha = np.zeros((obseq.shape[0], trans_log2.shape[0]))

    # alpha[0, :] = init_dist * emis[:, obseq[0]] # the use of obseq[] ensures the values in observation

    # alpha[0,:] calculation: log version
    for i in range(len(init_dist_log2)):
        alpha[0, i] = init_dist_log2[i] + emis_log2[i, obseq[0]]

    # sequence cannot be greater than the number of columns in the emission matrix
    # alpha[0, :] = init_dist * (emis[:, 0].T)

    for t in range(1, obseq.shape[0]):
        for j in range(trans_log2.shape[0]):
            # alpha[t, j] = alpha[t - 1].dot(trans[:, j]) * emis[j, obseq[t]] # this emis[j, obseq[t]] creates issue

            # Log operation of "alpha[t, j] = alpha[t - 1].dot(trans[:, j]) * emis[j, obseq[t]]"

            # reshape alpha and trans into 2D matrix from 1D
            alpha_transpose = np.reshape(alpha[t-1, :], (1, trans_log2.shape[0])) # transpose to multiple with trans_log[:,j]
            trans_log_temp = np.reshape(trans_log2[:, j], (trans_log2.shape[0], 1))

            # log version of "alpha[t - 1].dot(trans[:, j])"
            log_base = 2
            alpha_trans_prod_2d = log_space_product(alpha_transpose, trans_log_temp, log_base) # in 2D
            alpha_trans_prod_1d = alpha_trans_prod_2d.flatten() # convert 2D matrix into 1D matrix
            # print("\n Flattened: ", alpha_trans_prod_1d)
            # print("\n alpha_trans_dot_1d:", alpha_trans_prod_1d)

            alpha[t, j] = emis_log2[j, obseq[t]] + alpha_trans_prod_1d

            # the use of obseq[] ensures the values in observation sequence cannot be greater than
            # the number of columns in the emission matrix
    return alpha
