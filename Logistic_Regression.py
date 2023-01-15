from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
print((iris["target"] == 2).astype(np.int))
X = iris["data"]
Y = (iris["target"] == 2).astype(np.int)

clf = LogisticRegression()
clf.fit(X,Y)
print(clf.predict(iris["data"][:]))

