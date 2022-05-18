import numpy as np
from log_matrix_operations import log_space_product


# function definition of xi, the probability of being in two states x_i at t and x_j at t+1
def xi_estimation_procedure (obseq, trans_log2, emis_log2, alpha_variable, beta_variable, timestep):

    # define the xi matrix
    xi = np.zeros((trans_log2.shape[0], trans_log2.shape[0], timestep-1))
    # xi = np.zeros((timestep-1, trans.shape[0], trans.shape[0]))

    for t in range(timestep-1):
        # original version
        # denominator = np.dot(np.dot(alpha_variable[t, :].T, trans) * emis[:, obseq[t + 1]].T,beta_variable[t + 1, :])

        # log version of the calculation of "denominator"
        log_base = 2

        # reshape alpha_variable[t,:].T from (7,) into (1,7)
        # alpha_variable_temp = alpha_variable[t,:].reshape(trans.shape[0],1)
        alpha_variable_temp = alpha_variable[t,:].transpose() # shape = (7,)
        alpha_variable_temp = alpha_variable_temp[None, :] # shape = (1,7)

        # log version of "np.dot(alpha_variable[t, :].T, trans)"
        # alpha_trans_prod = log_space_product(alpha_variable_temp, trans_log) # create same rows always
        # log_space_product(alpha_variable_temp, trans_log) creates same rows always (just take one row)
        alpha_trans_prod = log_space_product(alpha_variable_temp, trans_log2, log_base)[0,:] # shape = (7,)
        alpha_trans_prod = alpha_trans_prod[None,:] # shape = (1,7)

        # reshape emis_log[:, obseq[t+1]].T from (7,) into (1,7)
        # emis_log_temp = emis_log[:, obseq[t+1]].reshape(1, trans.shape[0])
        emis_log2_temp = emis_log2[:, obseq[t+1]].transpose() # shape = (7,)
        emis_log2_temp = emis_log2_temp[None,:] # shape = (1,7)

        # log version of "np.dot(alpha_variable[t, :].T, trans) * emis[:, obseq[t + 1]].T"
        alpha_trans_prod_emis_mult = alpha_trans_prod + emis_log2_temp

        # reshape beta_variable[t+1,:] from (7,) into (7,7)
        # beta_variable_temp = beta_variable[t+1,:].reshape(trans.shape[0], 1)
        beta_variable_temp = beta_variable[t+1,:] # shape = (7,)
        beta_variable_temp = beta_variable_temp[:, None] # shape = (7,1)

        # denominator calculation
        denominator = log_space_product(alpha_trans_prod_emis_mult, beta_variable_temp, log_base) # shape = (7,7)

        # print("\n denominator: \n", denominator)

        for i in range(trans_log2.shape[0]):
            # original version
            # numerator = alpha_variable[t, i] * trans[i, :] * emis[:, obseq[t + 1]].T * beta_variable[t + 1, :].T

            # log version

            # reshape "trans_log[i,:]" from (7,) into (1,7)
            # trans_log_temp = trans_log[i,:].reshape(1,trans_log.shape[0])
            trans_log2_temp = trans_log2[i,:] # shape = (7,)
            trans_log2_temp = trans_log2_temp[None,:] # shape = (1,7)

            # reshape "beta_variable[t+1, :].T" from (7,) into (7,1)
            # beta_variable_temp_tplus1 = beta_variable[t+1,:].reshape(trans.shape[0],1)
            beta_variable_temp_tplus1 = beta_variable[t+1,:].transpose() # shape = (7,)
            beta_variable_temp_tplus1 = beta_variable_temp_tplus1[None,:] # shape = (1,7)

            # calculate "numerator"
            numerator = alpha_variable[t,i] + trans_log2_temp + emis_log2_temp + beta_variable_temp_tplus1 # shape = (7,7)

            # print("\n numerator: \n", numerator)

            # log version of "xi[i,:,t] = numerator / denominator"s
            xi[i,:,t] = numerator - denominator

            # xi[i,:,t] = numerator / denominator # original version

    return xi
