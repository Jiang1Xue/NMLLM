# -*- coding：utf-8 -*-

import numpy as np
import pandas as pd
import os
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import time
import random

def kNN(k, data):

    r, c = data.shape
    top_k = np.zeros((c, c))

    for i in range(c):
        print("KNN", i)
        x = data[:, i]
        distances = []
        for j in range(c):
            y = data[:, j]
            num = 0
            for m in range(len(y)):
                num = num + (x[m] - y[m]) ** 2
            num = sqrt(num)
            distances.append(num)
        distances = np.array(distances)
        nearest = np.argsort(distances)
        for l in nearest[:k]:
            top_k[i, l] = 1

    W = top_k
    print(sum(W))

    # 正则化adjacency matrix
    for i in range(c):
        num = sum(W[i, :]) - 1
        for j in range(c):
            if j == i:
                W[i,j] = 0
            else:
                W[i,j] = W[i,j]/num

    print(sum(W))

    return W

def compute_F(label_matrix, W, p):

    r, c = label_matrix.shape

    mid_1 = p * np.eye(r)
    mid_2 = np.dot(mid_1, W)
    mid_3 = np.eye(r) - mid_2
    mid_4 = np.linalg.inv(mid_3)

    mid_5 = np.eye(r) - mid_1
    mid_6 = np.dot(mid_5, label_matrix)


    F = np.dot(mid_4, mid_6)

    r, c = F.shape
    update_F = np.zeros((r,c))
    for i in range(r):
        index = np.argsort(-F[i, :])
        for j in index[:2]:
            update_F[i, j] = 1

    return update_F

def sample_label_trans(sample_label_state, sample_label_cognitive):
# 第一列是ad的标签
# 第二列是nad的标签
# 第三列是认知得分1的标签
# 第四列是认知得分2的标签
# 第五列是认知得分3的标签
# 第六列是认知得分4的标签
# 第七列是认知得分5的标签
# 第八列是认知得分6的标签

    true_matrix = np.zeros((596, 8))

    for i in range(len(sample_label_state)):
        if sample_label_state[i] == 1:
            true_matrix[i, 0] = 1
        if sample_label_state[i] == 0:
            true_matrix[i, 1] = 1
        if sample_label_cognitive[i] == 0:
            true_matrix[i, 2] = 1
        if sample_label_cognitive[i] == 1:
            true_matrix[i, 3] = 1
        if sample_label_cognitive[i] == 2:
            true_matrix[i, 4] = 1
        if sample_label_cognitive[i] == 3:
            true_matrix[i, 5] = 1
        if sample_label_cognitive[i] == 4:
            true_matrix[i, 6] = 1
        if sample_label_cognitive[i] == 5:
            true_matrix[i, 7] = 1


    return true_matrix

def main():
    # 读入数据
    os.chdir('/Users/xuejiang/PycharmProjects/isofom_/data/')
    # ========= Step 1. 读入数据 ===========
    isoform_expression_df = pd.read_csv('filter_isoform_express.csv')
    isoform_expression = isoform_expression_df.as_matrix()
    isoform_expression = isoform_expression[:, 2:]
    isoform_name = isoform_expression[:, :2]
    # 对每一列数据进行归一化处理
    scaler = MinMaxScaler()
    isoform_express_scaled = scaler.fit_transform(isoform_expression)
    isoform_express_scaled_T = isoform_express_scaled.T
    print(isoform_express_scaled_T.shape)


    sample_label_df = pd.read_csv('sample_label.csv')
    sample_label = sample_label_df.as_matrix()
    sample_name = sample_label[:, 0]
    sample_label_state = sample_label[:, 1]
    sample_label_cognitive = sample_label[:, 2]

    true_label = sample_label_trans(sample_label_state, sample_label_cognitive)
    print(true_label.shape)

    start_time_all = time.clock()
    W = kNN(70, isoform_express_scaled)
    W_df = pd.DataFrame(data = W)
    W_df.to_csv('/Users/xuejiang/PycharmProjects/isofom_/result/MLL_filter/k70/W.csv')

    # W_df = pd.read_csv('/Users/xuejiang/PycharmProjects/isofom_/result/MLL_filter/k60/W.csv')
    # W = W_df.as_matrix()
    # W = W[:, 1:]

    ### 用是ad 或者不是 ad 的标签进行实验
    original_label = []
    predict_label = []

    total = [j for j in range(596)]

    inter = 0
    for i in range(5):
        y = random.sample(total, 100)
        y = np.array(y)
        x = np.delete(total, y, axis=None)
        print("inter:", inter)
        inter = inter + 1
        y_train = true_label[x]
        y_test = true_label[y]
        r, c = y_test.shape
        print(y_train.shape)
        print(y_test.shape)
        unknown_label = np.zeros((r, c))
        label = np.vstack((unknown_label, y_train))
        predict = compute_F(label, W, 0.5)
        predict_unknown = predict[:r, :]
        original_label.append(y_test)
        predict_label.append(predict_unknown)

    stop_time_all = time.clock()
    cost_all = stop_time_all - start_time_all

    original_label = np.array(original_label)
    original_label = np.reshape(original_label, (-1, 8))
    print(original_label.shape)
    predict_label = np.array(predict_label)
    predict_label = np.reshape(predict_label, (-1, 8))
    print(predict_label.shape)

    # 保存结果
    cost = [cost_all, cost_all]
    cost = np.array(cost)
    cost_df = pd.DataFrame(data=cost)
    cost_df.to_csv('/Users/xuejiang/PycharmProjects/isofom_/result/MLL_filter/k70/10/time_cost.csv')

    original_label_df = pd.DataFrame(data=original_label)
    predict_label_df = pd.DataFrame(data=predict_label)

    original_label_df.to_csv('/Users/xuejiang/PycharmProjects/isofom_/result/MLL_filter/k70/10/true_label.csv')
    predict_label_df.to_csv('/Users/xuejiang/PycharmProjects/isofom_/result/MLL_filter/k70/10/predict_label.csv')


if __name__ == '__main__':
    main()
