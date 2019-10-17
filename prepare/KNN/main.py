from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# 导入uci数据
iris = datasets.load_iris()

X = iris.data
y = iris.target

print(X)
print(y)

# 把数据分为训练数据、测试数据
# random_state=2019表示随机种子，相同随机种子产生相同分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2019)

# 构建KNN模型，K=3，并做训练
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)

correct = np.count_nonzero(clf.predict(X_test) == y_test)
print("correct: %d, all: %d" % (correct, len(y_test)))
print(accuracy_score(y_test, clf.predict(X_test)))
