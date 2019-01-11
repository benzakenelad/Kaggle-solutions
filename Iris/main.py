import numpy as np
import pandas as pd
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv("C:\\Users\\elad\\Desktop\\Kaggle\\Iris\\iris_data.csv")
data = np.array(data)
np.random.shuffle(data)

# setosa is 1
# versicolor is 2
# virginica is 3
length = data.shape[0]
for i in range(length):
    if data[i,4] == "setosa":
        data[i, 4] = 1
    elif data[i,4] == "versicolor":
        data[i, 4] = 2
    elif data[i, 4] == "virginica":
        data[i, 4] = 3
    else:
        print("error in flower class")


X_train = data[:120, :4]
Y_train = data[:120, 4:]
X_test = data[120:, :4]
Y_test = data[120:, 4:]

L = []
for i in range(len(Y_train)):
    L.append(Y_train[i,0])
Y_train = np.array(L)

L = []
for i in range(len(Y_test)):
    L.append(Y_test[i,0])
Y_test = np.array(L)

# model

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
acc_log = logreg.score(X_train, Y_train)
print("Logistic Regression Train accuracy :  " + str(acc_log))
acc_log = logreg.score(X_test, Y_test)
print("Logistic Regression Test accuracy :  " + str(acc_log))



# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
acc_svc = svc.score(X_train, Y_train)
print("Support Vector Machines Train accuracy :  " + str(acc_svc))
acc_svc = svc.score(X_test, Y_test)
print("Support Vector Machines Test accuracy :  " + str(acc_svc))

# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
acc_knn = knn.score(X_train, Y_train)
print("KNN Train accuracy :  " +  str(acc_knn))
acc_knn = knn.score(X_test, Y_test)
print("KNN Test accuracy :  " +  str(acc_knn))


# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
acc_gaussian = gaussian.score(X_train, Y_train)
print("Gaussian Naive Bayes Train accuracy :  " + str(acc_gaussian))
acc_gaussian = gaussian.score(X_test, Y_test)
print("Gaussian Naive Bayes Test accuracy :  " + str(acc_gaussian))


# Perceptron
perceptron = Perceptron(max_iter=10000)
perceptron.fit(X_train, Y_train)
acc_perceptron = perceptron.score(X_train, Y_train)
print("Perceptron Train accuracy :  " + str(acc_perceptron))
acc_perceptron = perceptron.score(X_test, Y_test)
print("Perceptron Test accuracy :  " + str(acc_perceptron))


# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
acc_linear_svc = linear_svc.score(X_train, Y_train)
print("Linear SVC Train accuracy :  " + str(acc_linear_svc))
acc_linear_svc = linear_svc.score(X_test, Y_test)
print("Linear SVC Test accuracy :  " + str(acc_linear_svc))


# Stochastic Gradient Descent
sgd = SGDClassifier(max_iter=1000)
sgd.fit(X_train, Y_train)
acc_sgd = sgd.score(X_train, Y_train)
print("Stochastic Gradient Descent Train accuracy :  " + str(acc_sgd))
acc_sgd = sgd.score(X_test, Y_test)
print("Stochastic Gradient Descent Test accuracy :  " + str(acc_sgd))


# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
acc_decision_tree = decision_tree.score(X_train, Y_train)
print("Decision Tree Train accuracy :  " + str(acc_decision_tree))
acc_decision_tree = decision_tree.score(X_test, Y_test)
print("Decision Tree Test accuracy :  " + str(acc_decision_tree))


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
acc_random_forest = random_forest.score(X_train, Y_train)
print("Random Forest Train accuracy :  " + str(acc_random_forest))
acc_random_forest = random_forest.score(X_test, Y_test)
print("Random Forest Test accuracy :  " + str(acc_random_forest))
