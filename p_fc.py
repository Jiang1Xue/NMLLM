import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

def statistics_t(data_1, data_2):
    """
    统计基因在正常样本与疾病样本下的差异表达显著性p值
    :param data_1:
    :param data_2:
    """
    normal_express = data_1
    case_express = data_2

    (rn, cn) = np.array(normal_express).shape
    (rc, cc) = np.array(case_express).shape

    p = []
    for i in range(rn):
        if sum(normal_express[i, :]) * sum(case_express[i, :]) != 0:
            t_result = ttest_ind(normal_express[i, :], case_express[i, :], equal_var = False)
            p.append(t_result[1])
        else:
            p.append(1)

    p = np.array(p).reshape((len(p), 1))

    return p

def statistics_fc(data_1, data_2):
    """
    统计基因在正常样本与疾病样本下的差异表达显著性p值
    :param data_1:
    :param data_2:
    """
    name = data_1[:, :3]
    normal_matrix = data_1[:, 3:]
    case_matrix = data_2[:, 3:]
    (rn, cn) = np.array(normal_matrix).shape
    (rc, cc) = np.array(case_matrix).shape

    fc = []
    for i in range(rn):
        fc_gene = 0
        num = 0
        for m in range(normal_matrix.shape[1]):
            for n in range(case_matrix.shape[1]):
                if case_matrix[i, n] != 0:
                    fc_gene += normal_matrix[i, m]/case_matrix[i, n]
                    num += num
        if num != 0:
            fc_gene = fc_gene/num

        fc.append(fc_gene)

    fc = np.array(fc).reshape(len(fc), 1)
    name_fc = np.hstack((name, fc))

    return name_fc

def select_same_items(matrix1, matrix2, label):
    select_items = matrix2[:, :3]
    r, c = select_items.shape
    print(select_items.shape)
    for i in range(len(matrix1)):
        if matrix1[i, 1] == label:
            num = np.array(matrix2[:, i+3]).reshape(r, 1)
            select_items = np.hstack((select_items, num))

    select_items = np.array(select_items)
    return select_items


def main():

    #==========Step1.读入数据======================
    # isoform的表达数据
    isoform_expression_df = pd.read_csv('E:/Experiment/AD_2/preprocess_data/select_marker_isoform_express.csv')
    isoform_expression = isoform_expression_df.as_matrix()
    name = isoform_expression[:, :3]

    clinical_label_df = pd.read_csv('E:/Experiment/AD_2/preprocess_data/clinical_label.csv')
    clinical_label = clinical_label_df.as_matrix()

    # ========== Step2. 统计基因差异显著水平 ======================
    ad1 = select_same_items(clinical_label, isoform_expression, "1_1")
    nd1 = select_same_items(clinical_label, isoform_expression, "2_1")

    ad2 = select_same_items(clinical_label, isoform_expression, "1_2")
    nd2 = select_same_items(clinical_label, isoform_expression, "2_2")

    ad3 = select_same_items(clinical_label, isoform_expression, "1_3")
    nd3 = select_same_items(clinical_label, isoform_expression, "2_3")

    ad4 = select_same_items(clinical_label, isoform_expression, "1_4")
    nd4 = select_same_items(clinical_label, isoform_expression, "2_4")

    ad5 = select_same_items(clinical_label, isoform_expression, "1_5")
    nd5 = select_same_items(clinical_label, isoform_expression, "2_5")

    ad6 = select_same_items(clinical_label, isoform_expression, "1_6")
    nd6 = select_same_items(clinical_label, isoform_expression, "2_6")

    print(ad1[13:16, :])
    print(nd2[13:16, :])

    ad1_fc_rank = statistics_fc(ad1[0:13, :], nd1[0:13, :])
    ad2_fc_rank = statistics_fc(ad2[13:16, :], nd2[13:16, :])
    ad3_fc_rank = statistics_fc(ad3[16:20, :], nd3[16:20, :])
    ad4_fc_rank = statistics_fc(ad4[30:36, :], nd4[30:36, :])
    ad5_fc_rank = statistics_fc(ad5[36:51, :], nd4[36:51, :])
    ad6_fc_rank = statistics_fc(ad6[51:67, :], nd6[51:67, :])

    nd1_fc_rank = statistics_fc(ad1[67:146, :], nd1[67:146, :])
    nd2_fc_rank = statistics_fc(ad2[146:212, :], nd2[146:212, :])
    nd3_fc_rank = statistics_fc(ad3[213:256, :], nd3[213:256, :])
    nd4_fc_rank = statistics_fc(ad4[257:266, :], nd4[257:266, :])
    nd5_fc_rank = statistics_fc(ad5[267:278, :], nd4[267:278, :])
    nd6_fc_rank = statistics_fc(ad6[279:292, :], nd6[279:292, :])

    fc = np.vstack((ad1_fc_rank, ad2_fc_rank))
    fc = np.vstack((fc, ad3_fc_rank))
    fc = np.vstack((fc, ad4_fc_rank))
    fc = np.vstack((fc, ad5_fc_rank))
    fc = np.vstack((fc, ad6_fc_rank))
    fc = np.vstack((fc, nd1_fc_rank))
    fc = np.vstack((fc, nd2_fc_rank))
    fc = np.vstack((fc, nd3_fc_rank))
    fc = np.vstack((fc, nd4_fc_rank))
    fc = np.vstack((fc, nd5_fc_rank))
    fc = np.vstack((fc, nd6_fc_rank))

    print(name.shape)
    print(fc.shape)

    # ========== Step4. 保存数据 ==========
    gene_fc_df = pd.DataFrame(data = fc)

    gene_fc_df.to_csv('E:/Experiment/AD_2/preprocess_data/marker_isoform_fc .csv')

if __name__ == '__main__':
    main()
