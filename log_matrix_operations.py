import numpy as np
from scipy.special import logsumexp


# method definition to get the actual values of 1D matrix from base-2 logarithms
def get_actual_values_from_log2_1d(input_log2_matrix1d):
    actual_values = np.zeros(input_log2_matrix1d.shape)

    for i in range(len(input_log2_matrix1d)):
        actual_values[i] = 2 ** input_log2_matrix1d[i]
    return actual_values

# method definition to get the actual values of 2D matrix from base-2 logarithms
def get_actual_values_from_log2_2d(input_log2_matrix2d):
    actual_values = np.zeros(input_log2_matrix2d.shape)

    for i in range(input_log2_matrix2d.shape[0]):
        for j in range(input_log2_matrix2d.shape[1]):
            actual_values[i][j] = 2 ** input_log2_matrix2d[i][j]
    return actual_values


# method definition to perform matrix multiplication
def log_space_product(A, B, base):
    c = np.log(base)
    Astack = np.stack([A]*A.shape[0]).transpose(2,1,0)
    Bstack = np.stack([B]*B.shape[1]).transpose(1,0,2)
    return (1/c) * logsumexp(c*(Astack+Bstack), axis=0)




# It works very well with not-so-big (<1000 rows) matrices
# def logdot(a, b):
#     max_a, max_b = np.max(a), np.max(b)
#     exp_a, exp_b = a - max_a, b - max_b
#     np.exp(exp_a, out=exp_a)
#     np.exp(exp_b, out=exp_b)
#     c = np.dot(exp_a, exp_b)
#     np.log(c, out=c)
#     c += max_a + max_b
#     return c


# A = np.array([[0.5, 0.5],
#      [0.6, 0.4]])
#
# B = np.array([[0.3, 0.7],
#      [0.8, 0.2]])
# #
# # # C_matmul = np.matmul(A, B)
# # # C_dot = A.dot(B)
# # #
# # # print("\n A: ", A)
# # # print("\n B: ", B)
# # # print("\n C_matmul: ", C_matmul)
# # # print("\n C_dot: ", C_dot)
# #
# A_log = np.log(A)
# B_log = np.log(B)
#
# print("\n Log(A): ", A_log)
# print("\n Log(B): ", B_log)
#
# A_frow = np.reshape(A_log[0,:], (1,2))
# print("\n A_frow: ", A_frow)
# print("\n A_frow (shape): ", A_frow.shape)
#
# B_fcol = np.reshape(B_log[:,0], (2,1))
# print("\n B_fcol: ", B_fcol)
# print("\n B_fcol (shape): ", B_fcol.shape)
#
# print("\n Product: ", log_space_product(A_frow, B_fcol))
# print("\n Product (exp):", np.exp(log_space_product(A_frow, B_fcol)))
#
# print("\n C_log: ", logdotexp(A_log, B_log))
# print("\n C_log_frow_fcol (log): ", logdotexp(A_frow, B_fcol))
# print("\n C_log_frow_fcol (exp): ", np.exp(logdotexp(A_frow, B_fcol)))
# # print("\n Log(C_matmul): ", np.log(C_matmul))