# ****************** Was not used finally *****************

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn
from scipy.interpolate import make_interp_spline

# read the eval datasets
l5 = pd.read_csv('length-based/v2/sorted/performance_eval_l5_sorted.csv')
l10 = pd.read_csv('length-based/v2/sorted/performance_eval_l10_sorted.csv')
l15 = pd.read_csv('length-based/v2/sorted/performance_eval_l15_sorted.csv')
l20 = pd.read_csv('length-based/v2/sorted/performance_eval_l20_sorted.csv')
l25 = pd.read_csv('length-based/v2/sorted/performance_eval_l25_sorted.csv')
l30 = pd.read_csv('length-based/v2/sorted/performance_eval_l30_sorted.csv')
l35 = pd.read_csv('length-based/v2/sorted/performance_eval_l35_sorted.csv')
l40 = pd.read_csv('length-based/v2/sorted/performance_eval_l40_sorted.csv')
l45 = pd.read_csv('length-based/v2/sorted/performance_eval_l45_sorted.csv')
l50 = pd.read_csv('length-based/v2/sorted/performance_eval_l50_sorted.csv')
l60 = pd.read_csv('length-based/v2/sorted/performance_eval_l60_sorted.csv')
l70 = pd.read_csv('length-based/v2/sorted/performance_eval_l70_sorted.csv')
l80 = pd.read_csv('length-based/v2/sorted/performance_eval_l80_sorted.csv')
l90 = pd.read_csv('length-based/v2/sorted/performance_eval_l90_sorted.csv')
l100 = pd.read_csv('length-based/v2/sorted/performance_eval_l100_sorted.csv')

# set the "window_size" series
windows = l5['window_size'].values
window_new = np.linspace(windows.min(), windows.max(), 300)
# length_float = length.astype(np.float32)

# fetch all the necessary columns of a certain length (length = 5)
precision_l5 = l5['precision'].values
recall_l5 = l5['recall'].values
fscore_l5 = l5['f-score'].values
unordered_score_l5 = l5['unordered_fscore'].values
lcs_len_l5 = l5['lcs_len'].values

# fetch all the necessary columns of a certain length (length = 10)
precision_l10 = l10['precision'].values
recall_l10 = l10['recall'].values
fscore_l10 = l10['f-score'].values
unordered_score_l10 = l10['unordered_fscore'].values
lcs_len_l10 = l10['lcs_len'].values

# fetch all the necessary columns of a certain length (length = 15)
precision_l15 = l15['precision'].values
recall_l15 = l15['recall'].values
fscore_l15 = l15['f-score'].values
unordered_score_l15 = l15['unordered_fscore'].values
lcs_len_l15 = l15['lcs_len'].values

# fetch all the necessary columns of a certain length (length = 20)
precision_l20 = l20['precision'].values
recall_l20 = l20['recall'].values
fscore_l20 = l20['f-score'].values
unordered_score_l20 = l20['unordered_fscore'].values
lcs_len_l20 = l20['lcs_len'].values

# fetch all the necessary columns of a certain length (length = 25)
precision_l25 = l25['precision'].values
recall_l25 = l25['recall'].values
fscore_l25 = l25['f-score'].values
unordered_score_l25 = l25['unordered_fscore'].values
lcs_len_l25 = l25['lcs_len'].values

# fetch all the necessary columns of a certain length (length = 30)
precision_l30 = l30['precision'].values
recall_l30 = l30['recall'].values
fscore_l30 = l30['f-score'].values
unordered_score_l30 = l30['unordered_fscore'].values
lcs_len_l30 = l30['lcs_len'].values

# fetch all the necessary columns of a certain length (length = 35)
precision_l35 = l35['precision'].values
recall_l35 = l35['recall'].values
fscore_l35 = l35['f-score'].values
unordered_score_l35 = l35['unordered_fscore'].values
lcs_len_l35 = l35['lcs_len'].values

# fetch all the necessary columns of a certain length (length = 40)
precision_l40 = l40['precision'].values
recall_l40 = l40['recall'].values
fscore_l40 = l40['f-score'].values
unordered_score_l40 = l40['unordered_fscore'].values
lcs_len_l40 = l40['lcs_len'].values

# fetch all the necessary columns of a certain length (length = 45)
precision_l45 = l45['precision'].values
recall_l45 = l45['recall'].values
fscore_l45 = l45['f-score'].values
unordered_score_l45 = l45['unordered_fscore'].values
lcs_len_l45 = l45['lcs_len'].values

# fetch all the necessary columns of a certain length (length = 50)
precision_l50 = l50['precision'].values
recall_l50 = l50['recall'].values
fscore_l50 = l50['f-score'].values
unordered_score_l50 = l50['unordered_fscore'].values
lcs_len_l50 = l50['lcs_len'].values

# fetch all the necessary columns of a certain length (length = 70)
precision_l70 = l70['precision'].values
recall_l70 = l70['recall'].values
fscore_l70 = l70['f-score'].values
unordered_score_l70 = l70['unordered_fscore'].values
lcs_len_l70 = l70['lcs_len'].values

# fetch all the necessary columns of a certain length (length = 90)
precision_l90 = l90['precision'].values
recall_l90 = l90['recall'].values
fscore_l90 = l90['f-score'].values
unordered_score_l90 = l90['unordered_fscore'].values
lcs_len_l90 = l90['lcs_len'].values

# fetch all the necessary columns of a certain length (length = 100)
precision_l100 = l100['precision'].values
recall_l100 = l100['recall'].values
fscore_l100 = l100['f-score'].values
unordered_score_l100 = l100['unordered_fscore'].values
lcs_len_l100 = l100['lcs_len'].values

# ************************************************************************
# ******************* plot the figure for "estimating time" **************
# ************************************************************************
# plt.figure()
#
# plt.suptitle('Estimating Time (in milliseconds) vs Length of the Event Sequence')
#
# plt.subplot(231)
# plt.plot(length, est_time_w20, color = 'red', label = "Window = 20", linewidth = 2)
# plt.xlabel('Length of Sequence')
# plt.ylabel('Estimating Time (in milliseconds)')
# # plt.title('Estimating Time vs Length of the Event Sequence')
# plt.legend()
# plt.grid(True)
#
# plt.subplot(232)
# plt.plot(length, est_time_w40, color = 'red', label = "Window = 40", linewidth = 2)
# plt.xlabel('Length of Sequence')
# plt.ylabel('Estimating Time (in milliseconds)')
# # plt.title('Estimating Time vs Length of the Event Sequence')
# plt.legend()
# plt.grid(True)
#
# plt.subplot(233)
# plt.plot(length, est_time_w60, color = 'red', label = "Window = 60", linewidth = 2)
# plt.xlabel('Length of Sequence')
# plt.ylabel('Estimating Time (in milliseconds)')
# # plt.title('Estimating Time vs Length of the Event Sequence')
# plt.legend()
# plt.grid(True)
#
# plt.subplot(234)
# plt.plot(length, est_time_w80, color = 'red', label = "Window = 80", linewidth = 2)
# plt.xlabel('Length of Sequence')
# plt.ylabel('Estimating Time (in milliseconds)')
# # plt.title('Estimating Time vs Length of the Event Sequence')
# plt.legend()
# plt.grid(True)
#
# plt.subplot(235)
# plt.plot(length, est_time_w100, color = 'red', label = "Window = 100", linewidth = 2)
# plt.xlabel('Length of Sequence')
# plt.ylabel('Estimating Time (in milliseconds)')
# # plt.title('Estimating Time vs Length of the Event Sequence')
# plt.legend()
# plt.grid(True)
#
# plt.subplot(236)
# plt.plot(length, est_time_w120, color = 'red', label = "Window = 120", linewidth = 2)
# plt.xlabel('Length of Sequence')
# plt.ylabel('Estimating Time (in milliseconds)')
# # plt.title('Estimating Time vs Length of the Event Sequence')
# plt.legend()
# plt.grid(True)
#
# plt.show()

# ************************************************************************
# ******************* plot the figure for "decoding time" **************
# ************************************************************************
# plt.figure()
#
# plt.suptitle('Decoding Time (in milliseconds) vs Length of the Event Sequence')
#
# plt.subplot(231)
# plt.plot(length, dcd_time_w20, color = 'red', label = "Window = 20", linewidth = 2)
# plt.xlabel('Length of Sequence')
# plt.ylabel('Decoding Time (in milliseconds)')
# plt.legend()
# plt.grid(True)
#
# plt.subplot(232)
# plt.plot(length, dcd_time_w40, color = 'red', label = "Window = 40", linewidth = 2)
# plt.xlabel('Length of Sequence')
# plt.ylabel('Decoding Time (in milliseconds)')
# plt.legend()
# plt.grid(True)
#
# plt.subplot(233)
# plt.plot(length, dcd_time_w60, color = 'red', label = "Window = 60", linewidth = 2)
# plt.xlabel('Length of Sequence')
# plt.ylabel('Decoding Time (in milliseconds)')
# plt.legend()
# plt.grid(True)
#
# plt.subplot(234)
# plt.plot(length, dcd_time_w80, color = 'red', label = "Window = 80", linewidth = 2)
# plt.xlabel('Length of Sequence')
# plt.ylabel('Decoding Time (in milliseconds)')
# plt.legend()
# plt.grid(True)
#
# plt.subplot(235)
# plt.plot(length, dcd_time_w100, color = 'red', label = "Window = 100", linewidth = 2)
# plt.xlabel('Length of Sequence')
# plt.ylabel('Decoding Time (in milliseconds)')
# plt.legend()
# plt.grid(True)
#
# plt.subplot(236)
# plt.plot(length, dcd_time_w120, color = 'red', label = "Window = 120", linewidth = 2)
# plt.xlabel('Length of Sequence')
# plt.ylabel('Decoding Time (in milliseconds)')
# plt.legend()
# plt.grid(True)
#
# plt.show()

# ************************************************************************
# ******************* plot the figure for "fscore" ***********************
# ************************************************************************

# interpolate precisions with window_new
precision_l10_temp = make_interp_spline(windows, precision_l10, k=3)
precision_l10_new = precision_l10_temp(window_new)

precision_l30_temp = make_interp_spline(windows, precision_l30, k=3)
precision_l30_new = precision_l30_temp(window_new)

precision_l50_temp = make_interp_spline(windows, precision_l50, k=3)
precision_l50_new = precision_l50_temp(window_new)

precision_l70_temp = make_interp_spline(windows, precision_l70, k=3)
precision_l70_new = precision_l70_temp(window_new)

precision_l90_temp = make_interp_spline(windows, precision_l90, k=3)
precision_l90_new = precision_l90_temp(window_new)



# interpolate fscores with window_new
fscore_l5_temp = make_interp_spline(windows, fscore_l5, k=3)
fscore_l5_new = fscore_l5_temp(window_new)

fscore_l10_temp = make_interp_spline(windows, fscore_l10, k=3)
fscore_l10_new = fscore_l10_temp(window_new)

fscore_l15_temp = make_interp_spline(windows, fscore_l15, k=3)
fscore_l15_new = fscore_l15_temp(window_new)

fscore_l20_temp = make_interp_spline(windows, fscore_l20, k=3)
fscore_l20_new = fscore_l20_temp(window_new)

fscore_l25_temp = make_interp_spline(windows, fscore_l25, k=3)
fscore_l25_new = fscore_l25_temp(window_new)

fscore_l30_temp = make_interp_spline(windows, fscore_l30, k=3)
fscore_l30_new = fscore_l30_temp(window_new)

fscore_l35_temp = make_interp_spline(windows, fscore_l35, k=3)
fscore_l35_new = fscore_l35_temp(window_new)

fscore_l40_temp = make_interp_spline(windows, fscore_l40, k=3)
fscore_l40_new = fscore_l40_temp(window_new)

fscore_l45_temp = make_interp_spline(windows, fscore_l45, k=3)
fscore_l45_new = fscore_l45_temp(window_new)

fscore_l50_temp = make_interp_spline(windows, fscore_l50, k=3)
fscore_l50_new = fscore_l50_temp(window_new)

fscore_l70_temp = make_interp_spline(windows, fscore_l70, k=3)
fscore_l70_new = fscore_l70_temp(window_new)

fscore_l90_temp = make_interp_spline(windows, fscore_l90, k=3)
fscore_l90_new = fscore_l90_temp(window_new)

fscore_l100_temp = make_interp_spline(windows, fscore_l100, k=3)
fscore_l100_new = fscore_l100_temp(window_new)


# interpolate unordered fscores with window_new
unordered_score_l5_temp = make_interp_spline(windows, unordered_score_l5, k=3)
unordered_score_l5_new = unordered_score_l5_temp(window_new)

unordered_score_l10_temp = make_interp_spline(windows, unordered_score_l10, k=3)
unordered_score_l10_new = unordered_score_l10_temp(window_new)

unordered_score_l15_temp = make_interp_spline(windows, unordered_score_l15, k=3)
unordered_score_l15_new = unordered_score_l15_temp(window_new)

unordered_score_l20_temp = make_interp_spline(windows, unordered_score_l20, k=3)
unordered_score_l20_new = unordered_score_l20_temp(window_new)

unordered_score_l25_temp = make_interp_spline(windows, unordered_score_l25, k=3)
unordered_score_l25_new = unordered_score_l25_temp(window_new)

unordered_score_l30_temp = make_interp_spline(windows, unordered_score_l30, k=3)
unordered_score_l30_new = unordered_score_l30_temp(window_new)

unordered_score_l35_temp = make_interp_spline(windows, unordered_score_l35, k=3)
unordered_score_l35_new = unordered_score_l35_temp(window_new)

unordered_score_l40_temp = make_interp_spline(windows, unordered_score_l40, k=3)
unordered_score_l40_new = unordered_score_l40_temp(window_new)

unordered_score_l45_temp = make_interp_spline(windows, unordered_score_l45, k=3)
unordered_score_l45_new = unordered_score_l45_temp(window_new)

unordered_score_l50_temp = make_interp_spline(windows, unordered_score_l50, k=3)
unordered_score_l50_new = unordered_score_l50_temp(window_new)

unordered_score_l70_temp = make_interp_spline(windows, unordered_score_l70, k=3)
unordered_score_l70_new = unordered_score_l70_temp(window_new)

unordered_score_l90_temp = make_interp_spline(windows, unordered_score_l90, k=3)
unordered_score_l90_new = unordered_score_l90_temp(window_new)

unordered_score_l100_temp = make_interp_spline(windows, unordered_score_l100, k=3)
unordered_score_l100_new = unordered_score_l100_temp(window_new)

plt.figure()

plt.suptitle('Accuracy Score vs Window Size')

plt.subplot(211)
# plt.plot(window_new, fscore_l5_new, color = 'red', label = "l5-ordered", linewidth = 2)
plt.plot(window_new, fscore_l10_new, color = 'red', label = "l10", linewidth = 2)
# plt.plot(window_new, fscore_l15_new, color = 'blue', label = "l115-ordered", linewidth = 1.5)
# plt.plot(window_new, fscore_l20_new, color = 'green', label = "l20-ordered", linewidth = 1.5)
# plt.plot(window_new, fscore_l25_new, color = 'magenta', label = "l25-ordered", linewidth = 2)
plt.plot(window_new, fscore_l30_new, color = 'blue', label = "l30", linewidth = 2)
# plt.plot(window_new, fscore_l40_new, color = 'magenta', label = "l40-ordered", linewidth = 1)
plt.plot(window_new, fscore_l50_new, color = 'orange', label = "l50", linewidth = 2)
plt.plot(window_new, fscore_l70_new, color = 'magenta', label = "l70", linewidth = 2)
plt.plot(window_new, fscore_l90_new, color = 'green', label = "l90", linewidth = 2)
# plt.plot(window_new, fscore_l100_new, color = 'orange', label = "l100-ordered", linewidth = 1.5)

# plt.plot(window, unordered_score_l10, color = 'red', label = "l10-unordered", linewidth = 2)
# plt.plot(windows, unordered_score_l10, color = 'red', label = "l10-unordered", linewidth = 2)
# plt.plot(windows, unordered_score_l20, color = 'blue', label = "l20-unordered", linewidth = 2)
# plt.plot(windows, unordered_score_l30, color = 'green', label = "l30-unordered", linewidth = 2)
# plt.plot(windows, unordered_score_l40, color = 'green', label = "l40", linewidth = 1)
# plt.plot(windows, unordered_score_l50, color = 'black', label = "l50", linewidth = 1)
plt.xlabel('Window Size')
plt.ylabel('Accuracy Score (Ordered)')
plt.legend()
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.3)
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.subplot(212)
# # plt.plot(windows, fscore_l10, color = 'red', label = "l10", linewidth = 1)
# plt.plot(windows, fscore_l20, color = 'red', label = "l20-ordered", linewidth = 2)
# # plt.plot(windows, fscore_l30, color = 'blue', label = "l30", linewidth = 1)
# # plt.plot(windows, fscore_l40, color = 'green', label = "l40", linewidth = 1)
# # plt.plot(windows, fscore_l50, color = 'black', label = "l50", linewidth = 1)

# plt.plot(window_new, unordered_score_l5_new, color = 'black', label = "l5_unordered", linewidth = 1)
plt.plot(window_new, unordered_score_l10_new, color = 'red', label = "l10", linewidth = 2)
# plt.plot(window_new, unordered_score_l15_new, color = 'blue', label = "l15_unordered", linewidth = 2)
# plt.plot(window_new, unordered_score_l20_new, color = 'green', label = "l20_unordered", linewidth = 1.5)
plt.plot(window_new, unordered_score_l30_new, color = 'blue', label = "l30", linewidth = 2)
plt.plot(window_new, unordered_score_l50_new, color = 'orange', label = "l50", linewidth = 2)
plt.plot(window_new, unordered_score_l70_new, color = 'magenta', label = "l70", linewidth = 2)
plt.plot(window_new, unordered_score_l90_new, color = 'green', label = "l90", linewidth = 2)
# plt.plot(window_new, unordered_score_l100_new, color = 'orange', label = "l100_unordered", linewidth = 1.5)
plt.xlabel('Window Size')
plt.ylabel('Accuracy Score (Order not maintained)')
plt.legend()
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.3)
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.show()

# **************************************************************************************
# *************************** Scatter: Fscore vs Unordered Accuracy Score **************
# **************************************************************************************
plt.figure()

plt.suptitle('Ordered Accuracy Score vs Unordered Accuracy Score')

plt.subplot(111)

plt.plot(window_new, fscore_l10_new, color = 'red', label = "l10-ordered", linewidth = 2)
plt.plot(window_new, unordered_score_l10_new, color = 'orange', label = "l10-unordered", linewidth = 2)
plt.plot(window_new, fscore_l30_new, color = 'blue', label = "l30-ordered", linewidth = 2)
plt.plot(window_new, unordered_score_l30_new, color = 'magenta', label = "l30-unordered", linewidth = 2)
plt.plot(window_new, fscore_l50_new, color = 'green', label = "l50-ordered", linewidth = 2)
plt.plot(window_new, unordered_score_l50_new, color = 'black', label = "l50-unordered", linewidth = 2)

plt.xlabel('Window Size')
plt.ylabel('Accuracy Score')
plt.legend()
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.3)
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()

# **************************************************************************************
# *************************** Scatter: Precision vs Window Size **************
# **************************************************************************************
plt.figure()

plt.suptitle('Positive Predictive Value (Precision) vs Window Size')

plt.subplot(111)

plt.plot(window_new, precision_l10_new, color = 'red', label = "l10", linewidth = 2)
plt.plot(window_new, precision_l30_new, color = 'orange', label = "l30", linewidth = 2)
plt.plot(window_new, precision_l50_new, color = 'magenta', label = "l50", linewidth = 2)
# plt.plot(window_new, precision_l70_new, color = 'blue', label = "l70", linewidth = 2)
plt.plot(window_new, precision_l70_new, color = 'green', label = "l90", linewidth = 2)


plt.xlabel('Window Size')
plt.ylabel('Positive Predictive Value (Precision)')
plt.legend()
# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.3)
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()


# **************************************************************************************
# ************** Scatter: Fscore vs Unordered Accuracy Score (altogether) **************
# **************************************************************************************
plt.figure()

plt.suptitle('Ordered Accuracy Score vs Unordered Accuracy Score (For All Windows)')

data = {'ordered_accuracy_score': [fscore_l10_new, fscore_l30_new, fscore_l50_new, fscore_l70_new, fscore_l90_new],
        'unordered_accuracy_score': [unordered_score_l10_new, unordered_score_l30_new, unordered_score_l50_new, unordered_score_l70_new, unordered_score_l90_new]}

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
