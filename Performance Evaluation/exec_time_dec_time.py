import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn

# read the eval datasets
# w10 = pd.read_csv('csvs/performance_eval_w10_sorted.csv')
w20 = pd.read_csv('csvs/performance_eval_w20_sorted.csv')
# w30 = pd.read_csv('csvs/performance_eval_w30_sorted.csv')
w40 = pd.read_csv('csvs/performance_eval_w40_sorted.csv')
# w50 = pd.read_csv('csvs/performance_eval_w50_sorted.csv')
w60 = pd.read_csv('csvs/performance_eval_w60_sorted.csv')
# w70 = pd.read_csv('csvs/performance_eval_w70_sorted.csv')
w80 = pd.read_csv('csvs/performance_eval_w80_sorted.csv')
# w90 = pd.read_csv('csvs/performance_eval_w90_sorted.csv')
w100 = pd.read_csv('csvs/performance_eval_w100_sorted.csv')
# w110 = pd.read_csv('csvs/performance_eval_w110_sorted.csv')
w120 = pd.read_csv('csvs/performance_eval_w120_sorted.csv')


# set the "length" series
length = w20['length'].values
# length_float = length.astype(np.float32)

# fetch all the necessary columns of a certain window (window = 10)
# est_time_w10 = w10['estimating_time'].values
# dcd_time_w10 = w10['decoding_time'].values
# # precision_w10 = w10['precision'].values
# # recall_w10 = w10['recall'].values
# fscore_w10 = w10['f-score'].values
# unordered_score_w10 = w10['unordered_fscore'].values
# lcs_len_w10 = w10['lcs_len'].values

# fetch all the necessary columns of a certain window (window = 20)
est_time_w20 = w20['estimating_time'].values
dcd_time_w20 = w20['decoding_time'].values
# precision_w20 = w20['precision'].values
# recall_w20 = w20['recall'].values
fscore_w20 = w20['f-score'].values
unordered_score_w20 = w20['unordered_fscore'].values
lcs_len_w20 = w20['lcs_len'].values

# # fetch all the necessary columns of a certain window (window = 30)
# est_time_w30 = w30['estimating_time'].values
# dcd_time_w30 = w30['decoding_time'].values
# # precision_w30 = w30['precision'].values
# # recall_w30 = w30['recall'].values
# fscore_w30 = w30['f-score'].values
# unordered_score_w30 = w30['unordered_fscore'].values
# lcs_len_w30 = w30['lcs_len'].values

# fetch all the necessary columns of a certain window (window = 40)
est_time_w40 = w40['estimating_time'].values
dcd_time_w40 = w40['decoding_time'].values
# precision_w40 = w40['precision'].values
# recall_w40 = w40['recall'].values
fscore_w40 = w40['f-score'].values
unordered_score_w40 = w40['unordered_fscore'].values
lcs_len_w40 = w40['lcs_len'].values

# # fetch all the necessary columns of a certain window (window = 50)
# est_time_w50 = w50['estimating_time'].values
# dcd_time_w50 = w50['decoding_time'].values
# # precision_w50 = w50['precision'].values
# # recall_w50 = w50['recall'].values
# fscore_w50 = w50['f-score'].values
# unordered_score_w50 = w50['unordered_fscore'].values
# lcs_len_w50 = w50['lcs_len'].values

# fetch all the necessary columns of a certain window (window = 60)
est_time_w60 = w60['estimating_time'].values
dcd_time_w60 = w60['decoding_time'].values
# precision_w60 = w60['precision'].values
# recall_w60 = w60['recall'].values
fscore_w60 = w60['f-score'].values
unordered_score_w60 = w60['unordered_fscore'].values
lcs_len_w60 = w60['lcs_len'].values

# # fetch all the necessary columns of a certain window (window = 70)
# est_time_w70 = w70['estimating_time'].values
# dcd_time_w70 = w70['decoding_time'].values
# # precision_w70 = w70['precision'].values
# # recall_w70 = w70['recall'].values
# fscore_w70 = w70['f-score'].values
# unordered_score_w70 = w70['unordered_fscore'].values
# lcs_len_w70 = w70['lcs_len'].values

# fetch all the necessary columns of a certain window (window = 80)
est_time_w80 = w80['estimating_time'].values
dcd_time_w80 = w80['decoding_time'].values
# precision_w80 = w80['precision'].values
# recall_w80 = w80['recall'].values
fscore_w80 = w80['f-score'].values
unordered_score_w80 = w80['unordered_fscore'].values
lcs_len_w80 = w80['lcs_len'].values

# fetch all the necessary columns of a certain window (window = 90)
# est_time_w90 = w90['estimating_time'].values
# dcd_time_w90 = w90['decoding_time'].values
# # precision_w90 = w90['precision'].values
# # recall_w90 = w90['recall'].values
# fscore_w90 = w90['f-score'].values
# unordered_score_w90 = w90['unordered_fscore'].values
# lcs_len_w90 = w90['lcs_len'].values

# fetch all the necessary columns of a certain window (window = 100)
est_time_w100 = w100['estimating_time'].values
dcd_time_w100 = w100['decoding_time'].values
# precision_w100 = w100['precision'].values
# recall_w100 = w100['recall'].values
fscore_w100 = w100['f-score'].values
unordered_score_w100 = w100['unordered_fscore'].values
lcs_len_w100 = w100['lcs_len'].values


# fetch all the necessary columns of a certain window (window = 120)
est_time_w120 = w120['estimating_time'].values
dcd_time_w120 = w120['decoding_time'].values
# precision_w120 = w120['precision'].values
# recall_w120 = w120['recall'].values
fscore_w120 = w120['f-score'].values
unordered_score_w120 = w120['unordered_fscore'].values
lcs_len_w120 = w120['lcs_len'].values

# ************************************************************************
# ******************* plot the figure for "estimating time" **************
# ************************************************************************
plt.figure()

plt.suptitle('Estimating Time (in milliseconds) vs Length of the Event Sequence')

plt.subplot(231)
plt.plot(length, est_time_w20, color = 'red', label = "Window = 20", linewidth = 2)
plt.xlabel('Length of Sequence')
plt.ylabel('Estimating Time (in milliseconds)')
# plt.title('Estimating Time vs Length of the Event Sequence')
plt.legend()
plt.grid(True)

plt.subplot(232)
plt.plot(length, est_time_w40, color = 'red', label = "Window = 40", linewidth = 2)
plt.xlabel('Length of Sequence')
plt.ylabel('Estimating Time (in milliseconds)')
# plt.title('Estimating Time vs Length of the Event Sequence')
plt.legend()
plt.grid(True)

plt.subplot(233)
plt.plot(length, est_time_w60, color = 'red', label = "Window = 60", linewidth = 2)
plt.xlabel('Length of Sequence')
plt.ylabel('Estimating Time (in milliseconds)')
# plt.title('Estimating Time vs Length of the Event Sequence')
plt.legend()
plt.grid(True)

plt.subplot(234)
plt.plot(length, est_time_w80, color = 'red', label = "Window = 80", linewidth = 2)
plt.xlabel('Length of Sequence')
plt.ylabel('Estimating Time (in milliseconds)')
# plt.title('Estimating Time vs Length of the Event Sequence')
plt.legend()
plt.grid(True)

plt.subplot(235)
plt.plot(length, est_time_w100, color = 'red', label = "Window = 100", linewidth = 2)
plt.xlabel('Length of Sequence')
plt.ylabel('Estimating Time (in milliseconds)')
# plt.title('Estimating Time vs Length of the Event Sequence')
plt.legend()
plt.grid(True)

plt.subplot(236)
plt.plot(length, est_time_w120, color = 'red', label = "Window = 120", linewidth = 2)
plt.xlabel('Length of Sequence')
plt.ylabel('Estimating Time (in milliseconds)')
# plt.title('Estimating Time vs Length of the Event Sequence')
plt.legend()
plt.grid(True)

plt.show()

# ************************************************************************
# ******************* plot the figure for "decoding time" **************
# ************************************************************************
plt.figure()

plt.suptitle('Decoding Time (in milliseconds) vs Length of the Event Sequence')

plt.subplot(231)
plt.plot(length, dcd_time_w20, color = 'red', label = "Window = 20", linewidth = 2)
plt.xlabel('Length of Sequence')
plt.ylabel('Decoding Time (in milliseconds)')
plt.legend()
plt.grid(True)

plt.subplot(232)
plt.plot(length, dcd_time_w40, color = 'red', label = "Window = 40", linewidth = 2)
plt.xlabel('Length of Sequence')
plt.ylabel('Decoding Time (in milliseconds)')
plt.legend()
plt.grid(True)

plt.subplot(233)
plt.plot(length, dcd_time_w60, color = 'red', label = "Window = 60", linewidth = 2)
plt.xlabel('Length of Sequence')
plt.ylabel('Decoding Time (in milliseconds)')
plt.legend()
plt.grid(True)

plt.subplot(234)
plt.plot(length, dcd_time_w80, color = 'red', label = "Window = 80", linewidth = 2)
plt.xlabel('Length of Sequence')
plt.ylabel('Decoding Time (in milliseconds)')
plt.legend()
plt.grid(True)

plt.subplot(235)
plt.plot(length, dcd_time_w100, color = 'red', label = "Window = 100", linewidth = 2)
plt.xlabel('Length of Sequence')
plt.ylabel('Decoding Time (in milliseconds)')
plt.legend()
plt.grid(True)

plt.subplot(236)
plt.plot(length, dcd_time_w120, color = 'red', label = "Window = 120", linewidth = 2)
plt.xlabel('Length of Sequence')
plt.ylabel('Decoding Time (in milliseconds)')
plt.legend()
plt.grid(True)

plt.show()

# ************************************************************************
# ******************* plot the figure for "fscore" ***********************
# ************************************************************************
plt.figure()

plt.suptitle('Accuracy Score vs Unordered Accuracy Score vs Length of the Event Sequence')

plt.subplot(231)
plt.plot(length, fscore_w20, color = 'red', label = "Ordered (Window = 20)", linewidth = 2)
plt.plot(length, unordered_score_w20, color = 'blue', label = "Unordered (Window = 20)", linewidth = 2)
plt.xlabel('Length of Sequence')
plt.ylabel('Accuracy Score')
plt.legend()
plt.grid(True)

plt.subplot(232)
plt.plot(length, fscore_w40, color = 'red', label = "Ordered (Window = 40)", linewidth = 2)
plt.plot(length, unordered_score_w40, color = 'blue', label = "Unordered (Window = 40)", linewidth = 2)
plt.xlabel('Length of Sequence')
plt.ylabel('Accuracy Score')
plt.legend()
plt.grid(True)

plt.subplot(233)
plt.plot(length, fscore_w60, color = 'red', label = "Ordered (Window = 60)", linewidth = 2)
plt.plot(length, unordered_score_w60, color = 'blue', label = "Unordered (Window = 60)", linewidth = 2)
plt.xlabel('Length of Sequence')
plt.ylabel('Accuracy Score')
plt.legend()
plt.grid(True)

plt.subplot(234)
plt.plot(length, fscore_w80, color = 'red', label = "Ordered (Window = 80)", linewidth = 2)
plt.plot(length, unordered_score_w80, color = 'blue', label = "Unordered (Window = 80)", linewidth = 2)
plt.xlabel('Length of Sequence')
plt.ylabel('Accuracy Score')
plt.legend()
plt.grid(True)

plt.subplot(235)
plt.plot(length, fscore_w100, color = 'red', label = "Ordered (Window = 100)", linewidth = 2)
plt.plot(length, unordered_score_w100, color = 'blue', label = "Unordered (Window = 100)", linewidth = 2)
plt.xlabel('Length of Sequence')
plt.ylabel('Accuracy Score')
plt.legend()
plt.grid(True)

plt.subplot(236)
plt.plot(length, fscore_w120, color = 'red', label = "Ordered (Window = 120)", linewidth = 2)
plt.plot(length, unordered_score_w120, color = 'blue', label = "Unordered (Window = 120)", linewidth = 2)
plt.xlabel('Length of Sequence')
plt.ylabel('Accuracy Score')
plt.legend()
plt.grid(True)

plt.show()

# **************************************************************************************
# *************************** Scatter: Fscore vs Unordered Accuracy Score **************
# **************************************************************************************
plt.figure()

plt.suptitle('Ordered Accuracy Score vs Unordered Accuracy Score')

plt.subplot(231)
plt.scatter(fscore_w20, unordered_score_w20, alpha=0.5, color='blue', label='window = 20')
plt.xlabel('Ordered F-score')
plt.ylabel('Unordered Accuracy Score')
plt.legend()
plt.grid(True)

plt.subplot(232)
plt.scatter(fscore_w40, unordered_score_w40, alpha=0.5, color='blue', label='window = 40')
plt.xlabel('Ordered F-score')
plt.ylabel('Unordered Accuracy Score')
plt.legend()
plt.grid(True)

plt.subplot(233)
plt.scatter(fscore_w60, unordered_score_w60, alpha=0.5, color='blue', label='window = 60')
plt.xlabel('Ordered F-score')
plt.ylabel('Unordered Accuracy Score')
plt.legend()
plt.grid(True)

plt.subplot(234)
plt.scatter(fscore_w80, unordered_score_w80, alpha=0.5, color='blue', label='window = 80')
plt.xlabel('Ordered F-score')
plt.ylabel('Unordered Accuracy Score')
plt.legend()
plt.grid(True)

plt.subplot(235)
plt.scatter(fscore_w100, unordered_score_w100, alpha=0.5, color='blue', label='window = 100')
plt.xlabel('Ordered F-score')
plt.ylabel('Unordered Accuracy Score')
plt.legend()
plt.grid(True)

plt.subplot(236)
plt.scatter(fscore_w120, unordered_score_w120, alpha=0.5, color='blue', label='window = 120')
plt.xlabel('Ordered F-score')
plt.ylabel('Unordered Accuracy Score')
plt.legend()
plt.grid(True)

plt.show()

# **************************************************************************************
# ************** Scatter: Fscore vs Unordered Accuracy Score (altogether) **************
# **************************************************************************************
plt.figure()

plt.suptitle('Ordered Accuracy Score vs Unordered Accuracy Score (For All Windows)')

data = {'ordered_accuracy_score': [fscore_w20, fscore_w40, fscore_w60, fscore_w80, fscore_w100, fscore_w120],
        'unordered_accuracy_score': [unordered_score_w20, unordered_score_w40, unordered_score_w60, unordered_score_w80, unordered_score_w100, unordered_score_w120]}

plt.scatter(data['ordered_accuracy_score'], data['unordered_accuracy_score'], alpha=0.5, color='blue')
plt.xlabel('Ordered F-score')
plt.ylabel('Unordered Accuracy Score')
plt.legend()
plt.grid(True)

plt.show()


# **************************************************************************************
# *************************** Bar Chart: Event Length vs LCS Length ********************
# **************************************************************************************
plt.figure()
plt.suptitle('Ratio of Length of the Longest Extracted Sequence and \n Length of the Event Sequence \n vs \n Length of  the Event Sequence')

plt.subplot(231)
plt.bar(length, (lcs_len_w20.astype(np.float) / length.astype(np.float)) * 100, width=0.8, align='center', label='window = 20')
plt.xlabel('Length of the Event Sequence')
plt.ylabel('Ratio (in %)')
plt.legend()
plt.grid(True)

plt.subplot(232)
plt.bar(length, (lcs_len_w40.astype(np.float) / length.astype(np.float)) * 100, width=0.8, align='center', label='window = 40')
plt.xlabel('Length of the Event Sequence')
plt.ylabel('Ratio (in %)')
plt.legend()
plt.grid(True)

plt.subplot(233)
plt.bar(length, (lcs_len_w60.astype(np.float) / length.astype(np.float)) * 100, width=0.8, align='center', label='window = 60')
plt.xlabel('Length of the Event Sequence')
plt.ylabel('Ratio (in %)')
plt.legend()
plt.grid(True)

plt.subplot(234)
plt.bar(length, (lcs_len_w80.astype(np.float) / length.astype(np.float)) * 100, width=0.8, align='center', label='window = 80')
plt.xlabel('Length of the Event Sequence')
plt.ylabel('Ratio (in %)')
plt.legend()
plt.grid(True)

plt.subplot(235)
plt.bar(length, (lcs_len_w100.astype(np.float) / length.astype(np.float)) * 100, width=0.8, align='center', label='window = 100')
plt.xlabel('Length of the Event Sequence')
plt.ylabel('Ratio (in %)')
plt.legend()
plt.grid(True)

plt.subplot(236)
plt.bar(length, (lcs_len_w120.astype(np.float) / length.astype(np.float)) * 100, width=0.8, align='center', label='window = 120')
plt.xlabel('Length of the Event Sequence')
plt.ylabel('Ratio (in %)')
plt.legend()
plt.grid(True)

plt.show()


# **************************************************************************************
# *************** Horizontal Bar Chart: Event Length vs LCS Length ********************
# **************************************************************************************
# plt.figure()
# plt.subplot(221)
#
# plt.rcdefaults()
# fig, ax = plt.subplots()
# ax.barh(length, lcs_len_w10, align='center')
# # ax.set_yticks(y_pos)
# # ax.set_yticklabels(people)
# ax.invert_yaxis()  # labels read top-to-bottom
# ax.set_xlabel('Length of the Longest Identical Sequence')
# ax.set_title('Event Length vs Length of the Longest Sequence')
#
# plt.legend()
# plt.grid(True)
# plt.show()

# **************************************************************************************
# ****************** Stacked Bar Plot: Fscore vs Unordered Accuracy Score **************
# **************************************************************************************
# plt.figure()
#
# X_AXIS = length
#
# index = pd.Index(X_AXIS, name='Accuracy Score Comparison')
#
# data = {'Ordered Accuracy Score': fscore_w10,
#         'Unordered Accuracy Score': unordered_score_w10}
#
# df = pd.DataFrame(data, index=index)
# ax = df.plot(kind='bar', stacked=True, figsize=(10, 6))
# ax.set_ylabel('foo')
# plt.legend(title='Accuracy Score Comparison', bbox_to_anchor=(1.0, 1), loc='upper left')
# # plt.savefig('stacked.png')  # if needed
# plt.show()
#

# ***************************************************************************************
# ********************** For Correlation Heatmap using Seaborn **************************
# ***************************************************************************************

# create a dataframe
# dframe = pd.DataFrame(data={'length': length, 'fscore_w1': fscore_w1, 'fscore_w2': fscore_w2, 'fscore_w3': fscore_w3, 'fscore_w4': fscore_w4, 'fscore_w5': fscore_w5})
# dframe = pd.DataFrame(data={'fscore_w25': fscore_w25, 'fscore_w50': fscore_w50, 'fscore_w75': fscore_w75, 'fscore_w100': fscore_w100, 'fscore_w125': fscore_w125, 'fscore_w150': fscore_w150, 'fscore_w175': fscore_w175, 'fscore_w200': fscore_w200})
# dframe = pd.read_csv("window_size_fscore.csv")

plt.suptitle('Correlation among Ordered Accuracy Scores')
# plt.suptitle('Correlation among Unordered Accuracy Scores')

# ordered_accuracy_list = list(fscore_w20) + list(fscore_w40) + list(fscore_w60) + list(fscore_w80) + list(fscore_w100) + list(fscore_w120)
# unordered_accuracy_list = list(unordered_score_w20) + list(unordered_score_w40) + list(unordered_score_w60)
# unordered_accuracy_list += list(unordered_score_w80) + list(unordered_score_w100) + list(unordered_score_w120)

dframe = pd.DataFrame(data={'fscore_w20': fscore_w20, 'fscore_w40': fscore_w40, 'fscore_w60': fscore_w60, 'fscore_w80': fscore_w80, 'fscore_w100': fscore_w100, 'fscore_w120': fscore_w120})
# dframe = pd.DataFrame(data={'fscore_w20': unordered_score_w20, 'fscore_w40': unordered_score_w40, 'fscore_w60': unordered_score_w60, 'fscore_w80': unordered_score_w80, 'fscore_w100': unordered_score_w100, 'fscore_w120': unordered_score_w120})

plt.figure(figsize=(5,5))
seaborn.heatmap(dframe.corr(), annot=True, cmap="Blues")
plt.show()
