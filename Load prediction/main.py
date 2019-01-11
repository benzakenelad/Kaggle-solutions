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

data = pd.read_csv("C:\\Users\\elad\\Desktop\\Kaggle\\Load prediction\\train.csv")
data = np.array(data)
np.random.shuffle(data)

data = np.delete(data, 0, 1)

print(data[1])