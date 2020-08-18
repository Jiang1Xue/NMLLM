import pandas as pd
import numpy as np
import os

def overlap(filter_isoform, isoform_expression):
    r, c = isoform_expression.shape
    m, n = filter_isoform.shape

    filter_isoform_expression = []

    for i in range(r):
        for j in range(m):
            if filter_isoform[j, 0] == isoform_expression[i, 0]:
                filter_isoform_expression.append(isoform_expression[i, :])
                break

    filter_isoform_expression = np.array(filter_isoform_expression)

    return filter_isoform_expression

def main():
    # 读入数据
    os.chdir('/Users/xuejiang/PycharmProjects/isofom_/data/')
    # ========= Step 1. 读入数据 ===========
    isoform_expression_df = pd.read_csv('select_isoform_express.csv')
    isoform_expression = isoform_expression_df.as_matrix()

    filter_isoform_df = pd.read_csv('filter_isoform_name.csv')
    filter_isoform = filter_isoform_df.as_matrix()

    filter_isoform_express = overlap(filter_isoform, isoform_expression)

    filter_isoform_express_df = pd.DataFrame(data=filter_isoform_express)

    filter_isoform_express_df.to_csv('/Users/xuejiang/PycharmProjects/isofom_/data/filter_isoform_express.csv')

if __name__ == '__main__':
    main()




