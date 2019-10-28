#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入库
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# 随机生成样本
np.random.seed(12)
num_observations = 5000
x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
x2 = np.random.multivariate_normal([0, 4], [[1, .75], [.75, 1]], num_observations)
print(x1.shape, x2.shape)
X = np.vstack((x1, x2)).astype(np.float32)
y = np.hstack((np.zeros(num_observations), np.ones(num_observations)))
print(X.shape, y.shape)


# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 计算log likelihood
def log_likelihood(px, py, pw, pb):
    # 提取正负样本的下标
    pos, neg = np.where(py == 1), np.where(py == 0)
    # 使用向量化运算
    pos_sum = np.sum(np.log(sigmoid(np.dot(px[pos], pw) + pb)))
    neg_sum = np.sum(np.log(1 - sigmoid(np.dot(px[neg], pw) + pb)))
    return -(pos_sum + neg_sum)


def logistic_regression(px, py, num_steps, learning_rate):
    lw, lb = np.zeros(px.shape[1]), 0
    for step in range(num_steps):
        start = time.time() * 1e6
        # 预测值与真实值之间的误差
        error = sigmoid(np.dot(px, lw) + lb) - y
        # 梯度计算
        grad_w = np.matmul(px.T, error)
        grad_b = np.sum(error)
        # 梯度更新
        lw = lw - learning_rate * grad_w
        lb = lb - learning_rate * grad_b
        end = time.time() * 1e6
        if step % 10000 == 0:
            print(log_likelihood(px, y, lw, lb), end - start)
    return lw, lb


w, b = logistic_regression(X, y, num_steps=100000, learning_rate=5e-4)
print("w, b: ", w / w[0], b / w[0])

clf = LogisticRegression(fit_intercept=True, C=1e15, solver='liblinear')
clf.fit(X, y)
print("clf: ", clf.coef_[0] / clf.coef_[0][0], clf.intercept_ / clf.coef_[0][0])


def predict(px, pw, pb):
    return - (pw[0] / pw[1]) * px - pb / pw[1]


# 数据可视化
plt.figure(figsize=(12, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=.4)

pltX = np.array([-4, 4])
plt.plot(pltX, predict(pltX, w, b), color='red')
plt.plot(pltX, predict(pltX, clf.coef_[0], clf.intercept_), color='green')
plt.show()
