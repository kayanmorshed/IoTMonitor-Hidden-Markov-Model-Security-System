import numpy as np
from forward_procedure_algorithm import forward_procedure
from backward_procedure_algorithm import backward_procedure
from probability_of_being_in_two_states_calculation import xi_estimation_procedure
from convergence_checking_algorithm import convergence_checker
from scipy.special import logsumexp
from log_matrix_operations import get_actual_values_from_log2_1d, get_actual_values_from_log2_2d


# ***********************************************************************************************************
# ************************************** Baum Welch Algorithm ***********************************************
# ***********************************************************************************************************

# define Baum Welch algorithm to estimate both the state transition and emission probabilities
def baum_welch(obseq, trans, emis, init_dist, n_iter, conv_thresh):

    # Take the logarithms of "trans", "emis", and "init_dist"
    trans_log2 = np.log2(trans)
    emis_log2 = np.log2(emis)
    init_dist_log2 = np.log2(init_dist)

    log_of_base = np.log(2)  # for ln(2)

    timestep = len(obseq) # it would be len(sensor_value) for the dataset
    timestep_to_convergence = 0

    for n in range(n_iter):
        # call forward_procedure() from forward_procedure_algorithm.py to calculate "alpha"
        alpha_log2 = forward_procedure(obseq, trans_log2, emis_log2, init_dist_log2)

        # call backward_procedure() from backward_procedure_algorithm.py to calculate "beta"
        beta_log2 = backward_procedure(obseq, trans_log2, emis_log2)

        # This is to test the convergence
        transition_before_update = get_actual_values_from_log2_2d(trans_log2) # trans_log2 is in log2 form
        emission_before_update = get_actual_values_from_log2_2d(emis_log2) # emis_log2 is in log2 form

        # ************************************************************************************
        # We don't need to calculate "delta" since delta can be derived from xi directly using
        # ************************** equation 38 from Rabiner's paper ************************
        # ************************************************************************************

        # estimate "xi" variable using "xi_estimation_procedure()" from "probability_of_being_in_two_states_calculation.py"
        # calculate "x_log" using "xi_estimation_procedure()" from "probability_of_being_in_two_states_calculation.py"
        xi_log2 = xi_estimation_procedure(obseq, trans_log2, emis_log2, alpha_log2, beta_log2, timestep)

        # *********************************************************************************************
        # ************************************ Log version works perfectly upto this ******************
        # *********************************************************************************************

        # calculate "delta" i.e. Pr(being in state x_i at time t) from "xi"
        # delta_log = logsumexp(xi_log, axis=1) # delta_t(i) = sum(xi_t(i,j)) (eqn-38 in Rabiner's paper)

        # delta_t(i) = sum(xi_t(i,j)) (eqn-38 in Rabiner's paper)
        delta_log2 = (1/log_of_base) * logsumexp(log_of_base * xi_log2, axis=1)

        # update initial distribution
        init_dist_log2 = delta_log2[:,0]

        # update transition_probability_matrix
        # trans = np.sum(xi_actual, 2) / np.sum(delta, axis=1).reshape((-1, 1)) # original version
        # log version
        sum_xi_dim_2 = (1/log_of_base) * logsumexp(log_of_base * xi_log2, 2)
        sum_delta_dim_1 = (1/log_of_base) * logsumexp(log_of_base * delta_log2, axis=1)
        sum_delta_dim_1 = np.reshape(sum_delta_dim_1, (-1,1))

        trans_log2 =  sum_xi_dim_2 - sum_delta_dim_1

        # print("\n shape of trans_log: ", trans_log.shape)

        # print("\n Iteration: ", n+1)
        # print("State Transition Matrix after update: \n", trans)

        # Add additional T'th element in delta
        # delta = np.hstack((delta_log, np.sum(xi_actual[:,:,timestep-2], axis=0).reshape((-1, 1)))) # original version
        # log version
        sum_xi_dim_timestep_dim_0 = (1/log_of_base) * logsumexp(log_of_base * xi_log2[:, :, timestep - 2], axis=0)
        sum_xi_dim_timestep_dim_0 = np.reshape(sum_xi_dim_timestep_dim_0, (-1, 1))
        delta_log2 = np.hstack((delta_log2, sum_xi_dim_timestep_dim_0))

        K = emis_log2.shape[1]

        # denominator = np.sum(delta, axis=1) # original version
        # log version
        denominator = (1/log_of_base) * logsumexp(log_of_base * delta_log2, axis=1)

        for l in range(K):
            # emis[l, :] = emis[l, :] / denominator[l] # obselete version by sajid
            # this is changing the probabilities in EMIS_BEFORE_UPDATE matrix
            # emis[:, l] = np.sum(delta[:, obseq == l], axis=1)
            emis_log2[:, l] = (1/log_of_base) * logsumexp(log_of_base * delta_log2[:, obseq == l], axis=1)

        # update emission_probability_matrix
        # emis = np.divide(emis, denominator.reshape((-1, 1))) # original version
        # log version
        denominator_temp = denominator[:,None]
        emis_log2 = emis_log2 - denominator_temp # shape = (7,5) for states =7 and observations = 5
        # emis_log = np.reshape(emis_denominator_division, (-1,1)) # changes the shape from (7,5) to (35, 1)

        # print("\n shape of emis_log: ", emis_log.shape)

        # print("\n Emission Probability Matrix after update: \n", emis)
        # print("*****************************************************")

        convergence_checker_status = convergence_checker(transition_before_update, emission_before_update, get_actual_values_from_log2_2d(trans_log2), get_actual_values_from_log2_2d(emis_log2), conv_thresh)
        # print("\n Convergence Checker Status: ", convergence_checker_status)

        if convergence_checker_status == 1: # convergence_checker_status=1 means convergence occurs
            print("\n Converged at timestep: ", n)
            timestep_to_convergence = n
            break


    # ***********************************************************************************************
    # ******************* This conversion is causing problem in decoding algorithm ******************
    # ******* because we use np.log2() again in decoding algorithm to convert the probabilities *****
    # ***********************************************************************************************

    # convert the log versions of "trans", "emis", and "init_dist_log2" into actual ones
    # trans = get_actual_values_from_log2_2d(trans_log2) # trans_log2 is in log2 form
    # emis = get_actual_values_from_log2_2d(emis_log2) # emis_log2 is in log2 form
    # init_dist = get_actual_values_from_log2_1d(init_dist_log2) # emis_log2 is in log2 form

    # print("\n Trans (before): \n", trans)
    # print("\n Trans (After): \n", get_actual_values_from_log2_2d(trans_log2))
    # print("\n Emis (Before): \n", emis)
    # print("\n Emis (After): \n", get_actual_values_from_log2_2d(emis_log2))
    # print("\n Init Dist (Before): \n", init_dist)
    # print("\n Init Dist (After): \n", get_actual_values_from_log2_1d(init_dist_log2))

    # return (trans, emis, init_dist, timestep_to_convergence)
    return (trans_log2, emis_log2, init_dist_log2, timestep_to_convergence)
