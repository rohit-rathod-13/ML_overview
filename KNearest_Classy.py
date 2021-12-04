# KNearest_Classi.py
#_!!_#
#loading dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

iris = datasets.load_iris()
#print(iris.DESCR)

features = iris.data
labels = iris.target
print(features[0],labels[0])

#training the classifier
clf = KNeighborsClassifier()
clf.fit(features,labels)

#predicting the output
preds=clf.predict([[3,3,6,3]])
print(preds)


'''x=[[0],[1],[2],[3]]
y=[0,1,2,3]

clf=KNeighborsClassifier(n_neighbors =3)
clf.fit(x,y)

print(clf.predict([[0.1],[1.1],[1.5],[3.3],[4.5]]))

#To measure the accuracy
from sklearn.metrics import accuracy_score

y_pred =[0,1,2,3]
y_true =[0,1,2,3]

accuracy_score(y_true,y_pred)
'''
