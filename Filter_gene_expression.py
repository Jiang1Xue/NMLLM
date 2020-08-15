# -*- coding: utf-8 -*-
"""
作者：jx
日期:2019-09-26
版本：1
文件名：filter_gene_express.py
功能：根据样本过滤可用的基因，并过滤掉表达全为0的基因
"""
import numpy as np
import pandas as pd
import os

def filter_sample(expression_matrix, sample_matrix):
    rs = len(sample_matrix)
    r, c = expression_matrix.shape
    expression_filter = expression_matrix[:, :2]
    print(expression_filter.shape)

    for j in range(rs):
        for i in range(c):
            if expression_matrix[0, i] == sample_matrix[j]:
                middle_vector = np.array(expression_matrix[:, i]).reshape(r, 1)
                expression_filter = np.hstack((expression_filter, middle_vector))
                break

    expression_filter = np.array(expression_filter)
    rf, cf = expression_filter.shape
    print(rf)
    print(cf)

    col_name = np.array(sample_matrix[:]).reshape(1, rs)
    col_name_1 = np.array(['tracking id', 'gene id']).reshape(1, 2)
    col_name = np.hstack((col_name_1, col_name))
    expression_data = np.vstack((col_name, expression_filter))

    return expression_data

def filter_isoform(expression_matrix):
    # 一共有596个样本，如果一个基因中有550个样本的表达值全为0，则删除该基因
    filter_expression_matrix = expression_matrix[0, :]

    r, c = expression_matrix.shape
    for i in range(r):
        zero_number = 0
        middle_vector = expression_matrix[i, :]
        for j in range(len(middle_vector)):
            if middle_vector[j] == 0:
                zero_number += 1

        if zero_number < 50:
            print('zero number is %d for %d row'%(zero_number, i))
            filter_expression_matrix = np.vstack((filter_expression_matrix, middle_vector))

    rf, rc = filter_expression_matrix.shape
    print('the gene number before filtering is:', r)
    print('the gene number after filtering is:', rf)

    return filter_expression_matrix

def main():
    os.chdir('/Users/xuejiang/PycharmProjects/isofom_/data/')
    # ========= Step 1. 读入数据 ===========
    isoform_expression_df = pd.read_csv('ROSMAP_RNAseq_FPKM_isoform_normalized.csv')
    isoform_expression = isoform_expression_df.values

    sample_id_df = pd.read_csv('sample_clinical.csv')
    sample_id = sample_id_df.values
    sample_id = sample_id[:, 0]

    # 过滤样本
    isoform_expression_filter_sample = filter_sample(isoform_expression, sample_id)

    # 过滤基因
    isoform_expression_final = filter_isoform(isoform_expression_filter_sample)

    # =========== Step 2. 保存文件 ===========
    # 将Numpy.array格式转化为pandas.dataframe格式
    isoform_expression_filter_sample_df = pd.DataFrame(data = isoform_expression_filter_sample)
    isoform_expression_final_df = pd.DataFrame(data = isoform_expression_final)

    # 对文件进行输出
    #isoform_expression_filter_sample_df.to_csv('E:/Experiment/AD_2/preprocess_data/isoform_expression_filsamp.csv')
    isoform_expression_final_df.to_csv('isoform_expression_final.csv')

if __name__ == '__main__':
    main()