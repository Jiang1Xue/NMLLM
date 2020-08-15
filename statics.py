# -*- coding：utf-8 -*-

import numpy as np
import os
import pandas as pd

data_df = pd.read_csv('/Users/xuejiang/PycharmProjects/isofom_/result/MLL/k30/evaluation_result.csv')

data = data_df.values

ham_los_mean = np.mean(data[:, 1])
zero_one_los_mean = np.mean(data[:, 2])
coverage_error_mean = np.mean(data[:, 3])
label_ranking_loss_mean = np.mean(data[:, 4])
ave_precision_score_mean = np.mean(data[:, 5])

ham_los_std = np.std(data[:, 1])
zero_one_los_std = np.std(data[:, 2])
coverage_error_std = np.std(data[:, 3])
label_ranking_loss_std = np.std(data[:, 4])
ave_precision_score_std = np.std(data[:, 5])

print("ham_los_mean:", ham_los_mean)
print("ham_los_std", ham_los_std)
print("zero_one_los_mean", zero_one_los_mean)
print("zero_one_los_std", zero_one_los_std)
print("coverage_error_mean", coverage_error_mean)
print("coverage_error_std", coverage_error_std)
print("label_ranking_loss_mean", label_ranking_loss_mean)
print("label_ranking_loss_std", label_ranking_loss_std)
print("ave_precision_score_mean", ave_precision_score_mean)
print("ave_precision_score_std", ave_precision_score_std)