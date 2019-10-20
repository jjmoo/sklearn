#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# 导入iris数据
iris = datasets.load_iris()
X = iris.data
y = iris.target
print(X.shape, y.shape)

# 设置要搜索的k值
parameters = {'n_neighbors': [x for x in range(1, 16, 2)]}
knn = KNeighborsClassifier()

# 求最好的k值
clf = GridSearchCV(knn, parameters, cv=5)
clf.fit(X, y)

print("final best k: %d, score: %.2f" % (clf.best_params_['n_neighbors'], clf.best_score_))
