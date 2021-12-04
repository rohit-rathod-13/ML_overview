#Decision Tree
from sklearn import tree

x =[[0,0],[1,1]]
y =[0,1]

clf=tree.DecisionTreeClassifier()
clf=clf.fit(x,y)

clf.predict([[-2,-1]])

#array([0])

#Support Vector Machine

import numpy as np

x=np.array([[-1,-1],[-2,-1],[1,1],[2,1]])
y=np.array([1,1,2,2])

from sklearn.svm import SVC

clf=SVC()
clf.fit(x,y)

print(clf.predict([[8,1]]))
