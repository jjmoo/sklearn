#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

# 导入iris数据
iris = datasets.load_iris()
X = iris.data
y = iris.target
print(X.shape, y.shape)

# 定义候选的k值
ks = [x for x in range(1, 16, 2)]

# 5折交叉验证
kf = KFold(n_splits=5, random_state=2019, shuffle=True)

best_k = ks[0]
best_score = 0

for k in ks:
    curr_score = 0
    for train_index, valid_index in kf.split(X):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X[train_index], y[train_index])
        curr_score += clf.score(X[valid_index], y[valid_index])
    avg_score = curr_score / 5
    if best_score < avg_score:
        best_k = k
        best_score = avg_score
        print("current best score: %.2f, k: %d" % (best_score, best_k))

print("final best k: %d, score: %.2f" % (best_k, best_score))
