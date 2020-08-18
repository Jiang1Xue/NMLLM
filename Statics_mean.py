""""
作者：jx
日期：2019-11-12
版本：1
文件名：statics_mean.py
功能：统计每一个isofrom在596个样本中的均值和方差
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def pre_select_gene(data_df, mean, var):
    """
    1.计算每个isoform在所有样本中的均值和方差，并按从大到小进行排序
    2.画图可视化这些isoform的均值分布和方差分布
    3.筛选均值大于0.1，且方差大于1的isoform

    :param data_df:
    :param N
    :return: pre_select_gene
    """
    # 预处理基因表达数据
    #==============Step 1. 删除在所有样本中表达值全为0的基因==============
    data_matrix = data_df.values

    gename = data_matrix[:, :2]
    gene_express_matrix = data_matrix[:, 2:]

    # ===============Step 2. 计算每一个基因在所有样本中的方差和均值，并按从大到小进行排序=============
    proc_gene_express_matrix = np.array(gene_express_matrix)
    proc_gene_express_var = []
    proc_gene_express_mean = []

    # 对每一列数据进行归一化处理
    # scaler = MinMaxScaler()
    # gene_express_scaled = scaler.fit_transform(proc_gene_express_matrix)

    #计算每个基因在不同样本下的方差和均值，并按方差从大到小进行排名
    for i in range(len(gename)):
        proc_gene_express_var.append(proc_gene_express_matrix[i, :].var())
        proc_gene_express_mean.append(proc_gene_express_matrix[i, :].mean())

    #将list格式转化为数组格式，进而进行排序
    proc_var = np.array(proc_gene_express_var)
    proc_mean = np.array(proc_gene_express_mean)

    # 按从大到小进行排序，并记录排名对应的索引
    proc_var_sort = abs(np.sort(-proc_var))
    proc_var_sort_index = np.argsort(-proc_var)

    proc_mean_sort = abs(np.sort(-proc_mean))
    proc_mean_sort_index = np.argsort(-proc_mean)

    #作图，所有基因的方差和均值变化的曲线图
    plt.figure(1)
    plt.plot(proc_var_sort[:1000], 'k')
    plt.xlabel('Ranking')
    plt.ylabel('Variance')
    plt.title('Variances of isoform expression')
    plt.grid(True)
    plt.show()

    plt.figure(2)
    plt.plot(proc_mean_sort[:1000], 'k')
    plt.xlabel('Ranking')
    plt.ylabel('Mean')
    plt.title('Means of isoform expression')
    plt.grid(True)
    plt.show()

    #===============Step 3. 筛选高排名的基因 ================
    #筛选均值大于0.1，且方差大于1的isoform
    num = 0
    proc_gename = np.array(gename)
    proc1_gename = []
    proc1_gene_express_list = []

    print("length of gene list is", len(gename))
    print("length of variance is", len(proc_var))
    print("length of mean is", len(proc_mean))
    print("dimension of gene gene expression is", proc_gene_express_matrix.shape)

    for i in range(len(gename)):
        if proc_var[i] > var and proc_mean[i] > mean:
            num = num + 1
            proc1_gename.append(proc_gename[i, :])
            proc1_gene_express_list.append(proc_gene_express_matrix[i, :])

    print("the number of selected isoforms is", num)

    select_gename = np.array(proc1_gename)
    select_gene_express = np.array(proc1_gene_express_list)
    gename_gene_express = np.hstack((select_gename, select_gene_express))

    return gename_gene_express

def main():
    """
    主函数
    """
    os.chdir('/Users/xuejiang/PycharmProjects/isofom_/data/')
    # ========= Step 1. 读入数据，预筛选isoform ===========
    isoform_express_df = pd.read_csv('isoform_expression_final.csv')

    # 对数据初步预筛选基因
    select_isoform_express = pre_select_gene(isoform_express_df, 10, 1)

    # ======= Step 2. 保存文件 ============
    # 将Numpy.array格式转化为pandas.dataframe格式
    select_isoform_express_df = pd.DataFrame(data = select_isoform_express)

    # 对文件进行输出
    select_isoform_express_df.to_csv('select_isoform_express.csv')

if __name__ == '__main__':
    main()