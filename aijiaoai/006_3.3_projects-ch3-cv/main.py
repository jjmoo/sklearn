#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# 文件的读取，我们直接通过给定的`load_CIFAR10`模块读取数据。
# 感谢这个magic函数，你不必要担心如何写读取的过程。如果想了解细节，可以参考此文件。
from load_data import load_CIFAR10

import os
import numpy as np
import matplotlib.pyplot as plt

# 定义文件夹的路径：请不要修改此路径！ 不然提交后的模型不能够运行。
cifar10_dir = os.path.join(os.path.dirname(__file__), '../../data/cifar-10-batches-py')

# 读取文件，并把数据保存到训练集和测试集合。
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# 先来查看一下每个变量的大小，确保没有任何错误！X_train和X_test的大小应该为 N*W*H*3
# N: 样本个数, W: 样本宽度 H: 样本高度， 3: RGB颜色。 y_train和y_test为图片的标签。
print("训练数据和测试数据:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print("标签的种类: ", np.unique(y_train))  # 查看标签的个数以及标签种类，预计10个类别。
