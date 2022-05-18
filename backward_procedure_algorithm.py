import numpy as np
from log_matrix_operations import log_space_product


# **************************************************************************************************************
# *************** define the function backward procedure to calculate the probability of being *****************
# *************** at a certain state given all the future observation starting from this state *****************
# **************************************************************************************************************
def backward_procedure(obseq, trans_log2, emis_log2):
    beta = np.zeros((obseq.shape[0], trans_log2.shape[0]))
    # print("\n Shape of Beta: ", beta.shape)

    # setting beta(T) = 1
    # beta[obseq.shape[0] - 1] = np.ones((trans.shape[0]))
    beta[obseq.shape[0] - 1] = np.log(np.ones((trans_log2.shape[0]))) # Log version

    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(obseq.shape[0] - 2, -1, -1):
        for j in range(trans_log2.shape[0]):
            # beta[t, j] = (beta[t + 1] * emis[:, obseq[t + 1]]).dot(trans[j, :])

            # compute element-wise multiplication between beta[t+1] and emis[:,obseq[t+1]]
            beta_emis_element_wise_mult = np.zeros(trans_log2.shape[0])
            for i in range(trans_log2.shape[0]):
                beta_emis_element_wise_mult[i] = beta[t+1, i] + emis_log2[i, obseq[t+1]] # beta[t, j] already in log form;

            # reshape "beta_emis_element_wise_mult" into (1, trans.shape[0])
            beta_emis_element_wise_mult = np.reshape(beta_emis_element_wise_mult, (1, trans_log2.shape[0]))

            # reshape "trans_log[j,:]" into (trans.shape[0], 1)
            trans_log_ext = np.reshape(trans_log2[j,:], (trans_log2.shape[0],1))

            # dot multiplication between (beta[t+1] * emis[:,obseq[t+1]]) and (trans[j, :])
            log_base = 2
            beta[t, j] = log_space_product(beta_emis_element_wise_mult, trans_log_ext, log_base)
    return beta
