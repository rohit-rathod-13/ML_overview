import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import random

def studentReg(ages_train,net_worth_train):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(ages_train,net_worth_train)
    return reg

numpy.random.seed(42)

ages=[]

for ii in range(200):
    ages.append(random.randint(18,75))
    
net_worths = [ii * 6.25 +numpy.random.normal(scale=40) for ii in ages]

ages = numpy.reshape(numpy.array(ages),(len(ages),1))
net_worths = numpy.reshape(numpy.array(net_worths),(len(net_worths),1))

from sklearn.model_selection import train_test_split

ages_train,ages_test,net_worths_train,net_worths_test = train_test_split(ages,net_worths)

reg1 = studentReg(ages_train,net_worths_train)

print("Coefficient",reg1.coef_)
print("Intercept",reg1.intercept_)

print("training data",reg1.score(ages_train,net_worths_train))
print("Testing data",reg1.score(ages_test,net_worths_test))


plt.figure(figsize=(12,10))
sns.regplot(x=ages_train,y=net_worths_train,scatter=True,color="b",marker="*")
plt.xlabel("Ages Train")
plt.ylabel("Net Worth Train")
plt.title("Regression plot")


plt.figure(figsize=(12,10))
plt.scatter(ages_train,net_worths_train,color="b",label="train data")
plt.scatter(ages_test,net_worths_test,color="r",label="test data")
plt.plot(ages_test,reg1.predict(ages_test))

plt.xlabel("Ages")
plt.ylabel("Net Worth")
plt.legend(loc=2)
plt.show()

#Simple Linear Regression =(y=mx+c)
#Multiple Linear Regression=(y=c+ m1x1 +m..x..)
