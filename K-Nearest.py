from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()
# print(iris.DESCR)
features = iris.data
labels =iris.target

clf = KNeighborsClassifier()
clf.fit(features,labels)

preds =clf.predict([[1,1,1,1]])
if (preds==0):
    print("Iris-Setosa")
elif (preds==1):
    print("Iris-Versicolour")
else:
    print("Iris-Virginica")

  