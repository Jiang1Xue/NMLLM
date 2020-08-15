# -*- coding：utf-8 -*-

import numpy as np
import os
import pandas as pd

from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import average_precision_score

def read_data_mll(path):
    path_predict = path + os.sep + 'predict_label.csv'
    path_true = path + os.sep + 'true_label.csv'

    predict_df = pd.read_csv(path_predict)
    predict_label = predict_df.values
    predict_label = predict_label[:, 1:]
    true_df = pd.read_csv(path_true)
    true_label = true_df.values
    true_label = true_label[:, 1:]

    return predict_label, true_label


def read_data(path):
    path_s = path + os.sep + 'label_all_s.csv'
    path_c = path + os.sep + 'label_all_c.csv'

    data_s_df = pd.read_csv(path_s)
    data_s = data_s_df.values
    data_c_df = pd.read_csv(path_c)
    data_c = data_c_df.values

# 第一列是ad的标签
# 第二列是nad的标签
# 第三列是认知得分1的标签
# 第四列是认知得分2的标签
# 第五列是认知得分3的标签
# 第六列是认知得分4的标签
# 第七列是认知得分5的标签
# 第八列是认知得分6的标签
    predict_matrix = np.zeros((745, 8))
    true_matrix = np.zeros((745, 8))

    for i in range(1, len(data_s)):
        if data_s[i, 2] == '1':
            true_matrix[i-1, 0] = 1
        if data_s[i, 2] == '0':
            true_matrix[i-1, 1] = 1
        if data_c[i, 2] == '0':
            true_matrix[i-1, 2] = 1
        if data_c[i, 2] == '1':
            true_matrix[i-1, 3] = 1
        if data_c[i, 2] == '2':
            true_matrix[i-1, 4] = 1
        if data_c[i, 2] == '3':
            true_matrix[i-1, 5] = 1
        if data_c[i, 2] == '4':
            true_matrix[i-1, 6] = 1
        if data_c[i, 2] == '5':
            true_matrix[i-1, 7] = 1

    for i in range(1, len(data_s)):
        if data_s[i, 5] == '1':
            predict_matrix[i-1, 0] = 1
        if data_s[i, 5] == '0':
            predict_matrix[i-1, 1] = 1
        if data_c[i, 9] == '0':
            predict_matrix[i-1, 2] = 1
        if data_c[i, 9] == '1':
            predict_matrix[i-1, 3] = 1
        if data_c[i, 9] == '2':
            predict_matrix[i-1, 4] = 1
        if data_c[i, 9] == '3':
            predict_matrix[i-1, 5] = 1
        if data_c[i, 9] == '4':
            predict_matrix[i-1, 6] = 1
        if data_c[i, 9] == '5':
            predict_matrix[i-1, 7] = 1

    return true_matrix, predict_matrix

def compute_evaluation(true_matrix, predict_matrix):
    h = hamming_loss(true_matrix, predict_matrix)
    z = zero_one_loss(true_matrix, predict_matrix)
    c = coverage_error(true_matrix, predict_matrix) - 1
    r = label_ranking_loss(true_matrix, predict_matrix)
    a = average_precision_score(true_matrix, predict_matrix)

    result = [h, z, c, r, a]
    return result


def main():
    # 读入数据
    dir = '/Users/xuejiang/PycharmProjects/isofom_/result/MLL/k70'
    list = os.listdir(dir)
    print(list)

    result = []

    for i in range(len(list)):
        print(list[i])
        subject_code = list[i]
        path = '/Users/xuejiang/PycharmProjects/isofom_/result/MLL/k70/' + subject_code
        predict_matrix, true_matrix = read_data_mll(path)
        evaluation_result = compute_evaluation(true_matrix, predict_matrix)
        result.append(evaluation_result)

    result = np.array(result)

    # 保存结果
    result_df = pd.DataFrame(data=result)
    result_df.to_csv('/Users/xuejiang/PycharmProjects/isofom_/result/MLL/k70/evaluation_result.csv')

    print(result)

if __name__ == '__main__':
    main()