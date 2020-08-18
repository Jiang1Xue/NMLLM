# -*- coding：utf-8 -*-

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

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

def compute_H(m):

    H = np.zeros((596, 596))

    for i in range(596):
        for j in range(596):
            if i == j:
                H[i, j] = 1 - 1.0/596
            else:
                H[i, j] = - 1.0/596

    return H

def compute_XHYYHX(X, H, Y):

    X_t = X.T
    Y_t = Y.T

    result_1 = np.dot(X_t, H)
    result_2 = np.dot(result_1, Y)
    result_3 = np.dot(result_2, Y_t)
    result_4 = np.dot(result_3, H)
    result_5 = np.dot(result_4, X)

    return result_5

def select_plot(P):
    # 画出每一列（即每一列的置信性得分从大到小的排名）

    class_1 = P[:, 0]
    class_2 = P[:, 1]
    class_3 = P[:, 2]
    class_4 = P[:, 3]
    class_5 = P[:, 4]
    class_6 = P[:, 5]
    class_7 = P[:, 6]
    class_8 = P[:, 7]

    # 按从大到小进行排序
    class_1_sort = abs(np.sort(-abs(class_1)))
    class_2_sort = abs(np.sort(-abs(class_2)))
    class_3_sort = abs(np.sort(-abs(class_3)))
    class_4_sort = abs(np.sort(-abs(class_4)))
    class_5_sort = abs(np.sort(-abs(class_5)))
    class_6_sort = abs(np.sort(-abs(class_6)))
    class_7_sort = abs(np.sort(-abs(class_7)))
    class_8_sort = abs(np.sort(-abs(class_8)))

    # 作图，可视化置信得分的排名曲线图
    plt.figure(1)
    plt.subplot(241)
    plt.plot(class_1_sort[:1000], 'k')
    plt.xlabel('Ranking')
    plt.ylabel('Probability (confidence score)')
    plt.ylim(0, 0.1)
    plt.title('AD')


    plt.subplot(242)
    plt.plot(class_2_sort[:1000], 'k')
    plt.xlabel('Ranking')
    # plt.ylabel('Probability (confidence score)')
    plt.ylim(0, 0.1)
    plt.title('non-AD')


    plt.subplot(243)
    plt.plot(class_3_sort[:1000], 'k')
    plt.xlabel('Ranking')
    # plt.ylabel('Probability (confidence score)')
    plt.ylim(0, 0.1)
    plt.title('Cognitive score = 1')

    plt.subplot(244)
    plt.plot(class_4_sort[:1000], 'k')
    plt.xlabel('Ranking')
    # plt.ylabel('Probability (confidence score)')
    plt.ylim(0, 0.1)
    plt.title('Cognitive score = 2')


    plt.subplot(245)
    plt.plot(class_5_sort[:1000], 'k')
    plt.xlabel('Ranking')
    plt.ylabel('Probability (confidence score)')
    plt.ylim(0, 0.1)
    plt.title('Cognitive score = 3')


    plt.subplot(246)
    plt.plot(class_6_sort[:1000], 'k')
    plt.xlabel('Ranking')
    # plt.ylabel('Probability (confidence score)')
    plt.ylim(0, 0.1)
    plt.title('Cognitive score = 4')


    plt.subplot(247)
    plt.plot(class_7_sort[:1000], 'k')
    plt.xlabel('Ranking')
    # plt.ylabel('Probability (confidence score)')
    plt.ylim(0, 0.1)
    plt.title('Cognitive score = 5')


    plt.subplot(248)
    plt.plot(class_8_sort[:1000], 'k')
    plt.xlabel('Ranking')
    # plt.ylabel('Probability (confidence score)')
    plt.ylim(0, 0.1)
    plt.title('Cognitive score = 6')
    plt.show()

def select_key_gene(P, isoform_name):
    class_1 = P[:, 0]
    class_2 = P[:, 1]
    class_3 = P[:, 2]
    class_4 = P[:, 3]
    class_5 = P[:, 4]
    class_6 = P[:, 5]
    class_7 = P[:, 6]
    class_8 = P[:, 7]

    class_1_key_gene = []
    class_2_key_gene = []
    class_3_key_gene = []
    class_4_key_gene = []
    class_5_key_gene = []
    class_6_key_gene = []
    class_7_key_gene = []
    class_8_key_gene = []

    print("the length of confidence score", len(class_1))
    print("the length of isoform name", len(isoform_name))

    index_1 = np.argsort(-abs(class_1))
    index_2 = np.argsort(-abs(class_2))
    index_3 = np.argsort(-abs(class_3))
    index_4 = np.argsort(-abs(class_4))
    index_5 = np.argsort(-abs(class_5))
    index_6 = np.argsort(-abs(class_6))
    index_7 = np.argsort(-abs(class_7))
    index_8 = np.argsort(-abs(class_8))

    for i in index_1[:200]:
        class_1_key_gene.append(isoform_name[i])
    for i in index_2[:200]:
        class_2_key_gene.append(isoform_name[i])
    for i in index_3[:200]:
        class_3_key_gene.append(isoform_name[i])
    for i in index_4[:200]:
        class_4_key_gene.append(isoform_name[i])
    for i in index_5[:200]:
        class_5_key_gene.append(isoform_name[i])
    for i in index_6[:200]:
        class_6_key_gene.append(isoform_name[i])
    for i in index_7[:200]:
        class_7_key_gene.append(isoform_name[i])
    for i in index_8[:200]:
        class_8_key_gene.append(isoform_name[i])


    class_1_key_gene = np.array(class_1_key_gene)
    class_2_key_gene = np.array(class_2_key_gene)
    class_3_key_gene = np.array(class_3_key_gene)
    class_4_key_gene = np.array(class_4_key_gene)
    class_5_key_gene = np.array(class_5_key_gene)
    class_6_key_gene = np.array(class_6_key_gene)
    class_7_key_gene = np.array(class_7_key_gene)
    class_8_key_gene = np.array(class_8_key_gene)


    key_gene = np.hstack((class_1_key_gene, class_2_key_gene))
    key_gene = np.hstack((key_gene, class_3_key_gene))
    key_gene = np.hstack((key_gene, class_4_key_gene))
    key_gene = np.hstack((key_gene, class_5_key_gene))
    key_gene = np.hstack((key_gene, class_6_key_gene))
    key_gene = np.hstack((key_gene, class_7_key_gene))
    key_gene = np.hstack((key_gene, class_8_key_gene))
    col_names = ['ad key genes', 'ad key genes', 'nad key genes', 'nad key genes', 'cs1 key genes', 'cs1 key genes', 'cs2 key genes', 'cs2 key genes', 'cs3 key genes', 'cs3 key genes', 'cs4 key genes', 'cs4 key genes', 'cs5 key genes', 'cs5 key genes', 'cs6 key genes', 'cs6 key genes']
    col_names = np.array(col_names).reshape(1, len(col_names))
    key_gene = np.vstack((col_names, key_gene))

    return key_gene


def main():
    # 读入数据
    os.chdir('/Users/xuejiang/PycharmProjects/isofom_/data/')
    # ========= Step 1. 读入数据 ===========
    isoform_expression_df = pd.read_csv('select_isoform_express.csv')
    isoform_expression = isoform_expression_df.as_matrix()
    isoform_express = isoform_expression[:, 2:]
    isoform_express = isoform_express.T
    isoform_name = isoform_expression[:, :2]
    # 对每一列数据进行归一化处理
    scaler = MinMaxScaler()
    isoform_express_scaled = scaler.fit_transform(isoform_express)

    sample_label_df = pd.read_csv('sample_label.csv')
    sample_label = sample_label_df.as_matrix()
    sample_name = sample_label[:, 0]
    sample_label_state = sample_label[:, 1]
    sample_label_cognitive = sample_label[:, 2]

    true_label = sample_label_trans(sample_label_state, sample_label_cognitive)

    print(true_label.shape)
    print(sum(true_label))
    start_time_all = time.clock()
    H = compute_H(596)
    matrix = compute_XHYYHX(isoform_express_scaled, H, true_label)

    r,c = matrix.shape
    new_matrix = matrix
    for i in range(r):
        for j in range(c):
            new_matrix[i, j] = round(matrix[i,j], 5)

    eigenvalues, eigenvectors = np.linalg.eig(new_matrix)

    print(eigenvalues.shape)
    print(eigenvectors.shape)


    index = np.argsort(-eigenvalues)
    print(index)
    update_eigenvectors = []
    for i in index[:8]:
        update_eigenvectors.append(eigenvectors[:, i])
        print(eigenvalues[i])

    update_eigenvectors = np.array(update_eigenvectors)
    print(update_eigenvectors.shape)
    update_eigenvectors = update_eigenvectors.T
    print(update_eigenvectors.shape)

    select_plot(eigenvectors)

    key_genes = select_key_gene(update_eigenvectors, isoform_name)

    stop_time_all = time.clock()
    cost = stop_time_all - start_time_all

    # 保存结果
    cost = [cost, cost]
    cost = np.array(cost)
    cost_df = pd.DataFrame(data=cost)
    cost_df.to_csv('/Users/xuejiang/PycharmProjects/isofom_/result/MLL/time_cost_1.csv')

    key_genes_df = pd.DataFrame(data=key_genes)
    key_genes_df.to_csv('/Users/xuejiang/PycharmProjects/isofom_/result/MLL/key_genes.csv')

if __name__ == '__main__':
    main()