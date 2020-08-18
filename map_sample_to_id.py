#coding:utf-8
import pandas as pd
import numpy as np
import os

os.chdir('/Users/xuejiang/PycharmProjects/isofom_/data/ncov/')

sample_df = pd.read_excel('sample_id_c.xlsx')
sample = sample_df.values

sample_id_df = pd.read_excel('sample_id.xlsx')
sample_id = sample_id_df.values
# print(data.shape)

g = []
for i in range(len(sample[:, 0])):
    print(i)
    mid = 0
    for j in range(len(sample_id)):
        if sample[i, 0] == sample_id[j, 0]:
            mid = sample_id[j, 1]
            break
    g.append(mid)


g = np.array(g).reshape(len(g), 1)
print(g.shape)


final_label = np.hstack((sample, g))
final_label_df = pd.DataFrame(data = final_label)

# 对文件进行输出
final_label_df.to_csv('mapped_sample_id.csv')
