# Importing the packages and datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Loading the random dataset with 4 features among which 2 are informative
x,y =make_regression(n_samples=100,n_features=4,n_informative=2,random_state=0,shuffle = 'false')

x_train,x_test,y_train,y_test =train_test_split(x,y,random_state =0,shuffle ='false')

#  Creating object for forest Regressor
regr = RandomForestRegressor(max_depth=2,random_state =0)

# Training data
regr.fit(x_train,y_train)
pred_reg = regr.predict(x_test)
r2_score(y_test,pred_reg)


#Classification
# Loading the random dataset with 4 features among which 2 are informative
x1,y1 = make_classification(n_samples=1000,n_features=4,n_informative=2,n_redundant=0,random_state=0,shuffle='false')

x_train1,x_test1,y_train1,y_test1 =train_test_split(x1,y1,random_state=0,test_size=0.15)

#  Creating object for forest Classifier
clf=RandomForestClassifier(max_depth=2,random_state=0,criterion='entropy',min_samples_split=3)

#print(x1.shape)
#print(y1.shape)

#training the module
clf.fit(x_train1,y_train1)
pred_clf =clf.predict(x_test1)
accuracy_score(y_test1,pred_clf)

#Gives more accuracy with Bootstrap Aggregator
