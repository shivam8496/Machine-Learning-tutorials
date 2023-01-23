import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


cancer = datasets.load_breast_cancer()

x= cancer.data
y=cancer.target

x_train,x_test,y_train,y_test= sklearn.model_selection.train_test_split(x,y,test_size=0.2,random_state=42)

clf= svm.SVC(kernel='linear' , C=0.5 )
clf.fit(x_train,y_train)
pred= clf.predict(x_test)

acc = metrics.accuracy_score(y_test,pred)
print(acc)