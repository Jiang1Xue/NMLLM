# 导入所有的依赖包
import tensorflow as tf
import numpy as np
import pandas as pd
from CNNmodel import cnnModel
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
import sys
from sklearn.model_selection import StratifiedShuffleSplit
import os
import time

def trasform_data_format(data, label):
    r, c = data.shape

    # 创建空的多维数组用于存放数据
    dataset_array = np.zeros(shape = (c, 78, 78, 1))
    # 创建空的数组用于存放图片的标注信息
    dataset_labels = np.zeros(shape = (c), dtype=np.uint8)
    # 从文件夹下读取数据
    index = 0
    for i in range(c):
        # 格式转化为148x148x1shape
        data_reshaped = np.reshape(data[0:6084, i], newshape=(78, 78, 1))
        # 将维度转换后的图片存入指定的数组内
        dataset_array[index, :, :, :] = data_reshaped
        dataset_labels[index] = label[i]
        index = index + 1

    return dataset_array, dataset_labels


def create_model():
    # 判断是否有预训练模型
    model = cnnModel(0.5)
    model = model.createModel()
    return model

def main():
    # 读入数据
    os.chdir('/Users/xuejiang/PycharmProjects/isofom_/data/')
    # ========= Step 1. 读入数据 ===========
    isoform_expression_df = pd.read_csv('select_isoform_express.csv')
    isoform_expression = isoform_expression_df.as_matrix()
    isoform_expression = isoform_expression[:, 2:]
    isoform_name = isoform_expression[:, :2]
    # 对每一列数据进行归一化处理
    scaler = MinMaxScaler()
    isoform_express_scaled = scaler.fit_transform(isoform_expression)

    sample_label_df = pd.read_csv('sample_label.csv')
    sample_label = sample_label_df.as_matrix()
    sample_name = sample_label[:, 0]
    sample_label_state = sample_label[:, 1]
    sample_label_cognitive = sample_label[:, 2]

    isoform_express_scaled_state, sample_label_state = trasform_data_format(isoform_express_scaled, sample_label_state)
    # 对标签数据进行OneHot编码
    sample_label_state_onehot = tf.keras.utils.to_categorical(sample_label_state)

    isoform_express_scaled_cognitive, sample_label_cognitive = trasform_data_format(isoform_express_scaled,
                                                                                    sample_label_cognitive)
    # 对标签数据进行OneHot编码
    sample_label_cognitive_onehot = tf.keras.utils.to_categorical(sample_label_cognitive)

    # ### 用是ad 或者不是 ad 的标签进行实验
    # true_sample_name_s = []
    # true_label_s = []
    # predict_label_s = []
    #
    # ss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, train_size=0.75, random_state=0)
    #
    # start_time_all = time.clock()
    # for train_index, test_index in ss.split(isoform_express_scaled_state, sample_label_state_onehot):
    #     X_train, X_test = isoform_express_scaled_state[train_index], isoform_express_scaled_state[test_index]
    #     y_train, y_test = sample_label_state_onehot[train_index], sample_label_state_onehot[test_index]
    #     y_test_true = sample_label_state[test_index]
    #     sample_test_name = sample_name[test_index]
    #     model = create_model()
    #     model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=20, verbose=1)
    #     predict = model.predict(X_test)
    #     true_sample_name_s.append(sample_test_name)
    #     true_label_s.append(y_test_true)
    #     predict_label_s.append(predict)
    #
    # stop_time_all = time.clock()
    # cost_all_s = stop_time_all - start_time_all
    #
    # true_sample_name_s = np.array(true_sample_name_s)
    # true_sample_name_s = true_sample_name_s.flatten()
    # true_sample_name_s = true_sample_name_s.T
    # true_sample_name_s = np.reshape(true_sample_name_s, (len(true_sample_name_s), 1))
    # print(true_sample_name_s.shape)
    # true_label_s = np.array(true_label_s)
    # true_label_s = true_label_s.flatten()
    # true_label_s = true_label_s.T
    # true_label_s = np.reshape(true_label_s, (len(true_label_s), 1))
    # print(true_label_s.shape)
    # predict_label_s = np.array(predict_label_s)
    # predict_label_s = np.reshape(predict_label_s, (-1, 2))
    # print(predict_label_s.shape)
    #
    # final_pre_s = predict_label_s.argmax(axis=1)
    # final_pre_s = np.array(final_pre_s)
    # final_pre_s = np.reshape(final_pre_s, (len(final_pre_s), 1))
    # print(final_pre_s.shape)
    #
    # label_all_s = np.hstack((true_sample_name_s, true_label_s))
    # label_all_s = np.hstack((label_all_s, predict_label_s))
    # label_all_s = np.hstack((label_all_s, final_pre_s))
    # print(label_all_s.shape)
    # col_names = ['sample name', 'true_label', 'predict 0', 'predict 1', 'predict']
    # col_names = np.array(col_names)
    # label_all_s = np.vstack((col_names, label_all_s))

    ### 用认知评价得分进行实验
    true_sample_name_c = []
    true_label_c = []
    predict_label_c = []

    ss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, train_size=0.75, random_state=0)

    start_time_all = time.clock()
    for train_index, test_index in ss.split(isoform_express_scaled_cognitive, sample_label_cognitive_onehot):
        X_train, X_test = isoform_express_scaled_state[train_index], isoform_express_scaled_cognitive[test_index]
        y_train, y_test = sample_label_cognitive_onehot[train_index], sample_label_cognitive_onehot[test_index]
        y_test_true = sample_label_cognitive[test_index]
        sample_test_name = sample_name[test_index]
        model = create_model()
        model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=20, verbose=1)
        predict = model.predict(X_test)
        true_sample_name_c.append(sample_test_name)
        true_label_c.append(y_test_true)
        predict_label_c.append(predict)

    stop_time_all = time.clock()
    cost_all_c = stop_time_all - start_time_all

    true_sample_name_c = np.array(true_sample_name_c)
    true_sample_name_c = true_sample_name_c.flatten()
    true_sample_name_c = true_sample_name_c.T
    true_sample_name_c = np.reshape(true_sample_name_c, (len(true_sample_name_c), 1))
    print(true_sample_name_c.shape)
    true_label_c = np.array(true_label_c)
    true_label_c = true_label_c.flatten()
    true_label_c = true_label_c.T
    true_label_c = np.reshape(true_label_c, (len(true_label_c), 1))
    print(true_label_c.shape)
    predict_label_c = np.array(predict_label_c)
    predict_label_c = np.reshape(predict_label_c, (-1, 6))
    print(predict_label_c.shape)

    final_pre_c = predict_label_c.argmax(axis=1)
    final_pre_c = np.array(final_pre_c)
    final_pre_c = np.reshape(final_pre_c, (len(final_pre_c), 1))

    label_all_c = np.hstack((true_sample_name_c, true_label_c))
    label_all_c = np.hstack((label_all_c, predict_label_c))
    label_all_c = np.hstack((label_all_c, final_pre_c))
    col_names = ['sample name', 'true_label', 'predict 0', 'predict 1', 'predict 2', 'predict 3', 'predict 4',
                 'predict 5', 'predict']
    col_names = np.array(col_names)
    label_all_c = np.vstack((col_names, label_all_c))

    # 保存结果
    cost = [cost_all_c]
    cost = np.array(cost)
    cost_df = pd.DataFrame(data=cost)
    cost_df.to_csv('/Users/xuejiang/PycharmProjects/isofom_/result/cnn/5/time_cost_c.csv')

    # label_all_s_df = pd.DataFrame(data=label_all_s)
    label_all_c_df = pd.DataFrame(data=label_all_c)

    # label_all_s_df.to_csv('/Users/xuejiang/PycharmProjects/isofom_/result/cnn/9/label_all_s.csv')
    label_all_c_df.to_csv('/Users/xuejiang/PycharmProjects/isofom_/result/cnn/5/label_all_c.csv')


if __name__ == '__main__':
    main()



