import os
from scipy.io import loadmat
import numpy as np
import random
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
from collections import Counter
import matplotlib.pyplot as plt
# import smote_variants as sv


def iteror_raw_data(data_path, data_mark):
    """
       打标签，并返回数据的生成器：标签，样本数据。

       :param data_path：.mat文件所在路径
       :param data_mark："FE" 或 "DE"
       :return iteror：（标签，样本数据）
    """

    # 标签数字编码
    labels = {"98": 0, "106": 1, "119": 2, "131": 3, "108": 4,
              "118": 5, "110": 6, "270": 7, "271": 8, "272": 9, "273": 10}

    # 列出所有文件
    filenames = os.listdir(data_path)

    # 逐个对mat文件进行打标签和数据提取
    for single_txt in filenames:

        # 构建单个文件的完整路径
        single_mat_path = os.path.join(data_path, single_txt)

        # 打标签
        for key, _ in labels.items():
            if key in single_txt:
                label = labels[key]

        # 数据提取，从文件中加载文本数据
        data = np.loadtxt(single_mat_path)

        # 使用生成器逐个返回标签和对应的数据
        yield label, data


def data_augment(fs, win_tlen, overlap_rate, data_iteror):
    """
        此函数用于通过滑动窗口采样的方式对原始数据进行增强。

        :param fs: 原始数据的采样频率
        :param win_tlen: 滑动窗口的时间长度
        :param overlap_rate: 重叠部分比例, [0-100]，百分数；
        :param data_iteror: 原始数据的生成器格式
        :return (X, y): X 是切分好的数据， y 是数据标签
                        X[0].shape == (win_len,)
                        X.shape == (N, win_len)
    """
    # 将重叠率转换为整数类型
    overlap_rate = int(overlap_rate)
    # 计算窗口的长度，单位为采样点数
    win_len = int(fs * win_tlen)
    # 计算重合部分的时间长度，单位为采样点数
    overlap_len = int(overlap_rate // 100 * win_tlen * fs)
    # 计算步长，单位为采样点数
    step_len = int(win_len - overlap_len)

    # 初始化空列表，用于存储切分后的数据和对应的标签
    X = []
    y = []
    # 遍历原始数据生成器
    for iraw_data in data_iteror:
        # 获取单个原始数据并展平为一维数组
        single_raw_data = iraw_data[1].ravel()
        # 获取该数据对应的标签
        lab = iraw_data[0]
        # 获取单个原始数据的长度
        len_data = single_raw_data.shape[0]

        # 通过 zip 函数生成窗口的起始索引和结束索引
        for start_ind, end_ind in zip(range(0, len_data - win_len, step_len),
                                      range(win_len, len_data, step_len)):
            # 将窗口内的数据展平后添加到 X 列表中
            X.append(single_raw_data[start_ind:end_ind].ravel())
            # 将对应的标签添加到 y 列表中
            y.append(lab)

    # 将存储数据的列表转换为 NumPy 数组
    X = np.array(X)
    # 将存储标签的列表转换为 NumPy 数组
    y = np.array(y)

    # 返回切分好的数据和对应的标签
    return X, y


def under_sample_for_c0(X, y, low_c0, high_c0, random_seed):  # -> 没有使用
    """
    使用非0类别数据的数目，来对0类别数据进行降采样。

    :param X: 增强后的振动序列
    :param y: 类别标签0 - 9
    :param low_c0: 第一个类别0样本的索引下标
    :param high_c0: 最后一个类别0样本的索引下标
    :param random_seed: 随机种子
    :return X, y: 降采样后的振动序列和类别标签
    """

    # 设置随机种子，保证结果可复现
    np.random.seed(random_seed)
    # 计算需要从类别0样本中删除的索引
    # 目标是让类别0样本数量和类别3样本数量相同
    to_drop_ind = random.sample(
        range(low_c0, high_c0),
        (high_c0 - low_c0 + 1) - len(y[y == 3])  # 需要删除的样本数量
    )
    # 按照行删除，从 X 中移除指定索引对应的样本
    X = np.delete(X, to_drop_ind, 0)
    # 按照行删除，从 y 中移除指定索引对应的标签
    y = np.delete(y, to_drop_ind, 0)
    return X, y


def over_sample(X, y, len_data, random_seed=None):  # -> 没有使用
    """
    对样本较少的类别进行过采样，增加样本数目，实现样本平衡

    :param X: 特征矩阵，包含所有样本的特征
    :param y: 标签数组，对应每个样本的类别标签
    :param len_data: 数据的长度，不过在函数中未被使用
    :param random_seed: 随机种子，用于确保结果的可重复性，默认为 None
    :return: 过采样后的特征矩阵和标签数组
    """

    # 创建一个多类过采样器对象，使用 distance_SMOTE 算法进行过采样
    # sv 应该是某个库的别名，但此处未定义，需要检查导入情况
    oversampler = sv.MulticlassOversampling(
        # 设置 distance_SMOTE 算法的随机种子，用于复现结果
        sv.distance_SMOTE(random_state=random_seed)
    )
    # 调用过采样器的 sample 方法，对输入的特征矩阵 X 和标签数组 y 进行过采样
    # 返回过采样后的特征矩阵 X_samp 和标签数组 y_samp
    X_samp, y_samp = oversampler.sample(X, y)
    # 返回过采样后的特征矩阵和标签数组
    return X_samp, y_samp


def preprocess(path, data_mark, fs, win_tlen,
               overlap_rate, random_seed, **kargs):
    win_len = int(fs * win_tlen)
    data_iteror = iteror_raw_data(path, data_mark)
    X, y = data_augment(fs, win_tlen, overlap_rate, data_iteror, **kargs)
    # print(len(y[y==0]))

    print("-> 数据位置:{}".format(path))
    print("-> 原始数据采样频率:{0}Hz,\n-> 数据增强后共有：{1}条,"
          .format(fs, X.shape[0]))
    print("-> 单个数据长度：{0}采样点,\n-> 重叠量:{1}个采样点,"
          .format(X.shape[1], int(overlap_rate * win_tlen * fs // 100)))
    print("-> 类别数据数目:", sorted(Counter(y).items()))
    return X, y
