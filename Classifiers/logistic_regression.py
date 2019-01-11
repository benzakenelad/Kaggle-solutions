import numpy as np


class LogisticRegression:
    def __init__(self):
        self.theta = None

    def fit(self, X, y, eta=0.1, iterations=256):
        m = len(y)
        n = len(X[0])
        self.theta = np.random.normal(size=n)
        for i in range(iterations):
            gradient = X.transpose().dot(self.__sigmoid(X) - y)
            self.theta = self.theta - (eta / m) * gradient

        print('loss : {}'.format(self.__loss(X, y)))

    def predict(self, X):
        return self.__sigmoid(X)

    def __sigmoid(self, X):
        z = np.array(X.dot(self.theta), dtype=np.float32)
        return 1 / (1 + np.exp(-z))

    def __loss(self, X, y):
        h = self.__sigmoid(X)
        return -np.sum((2 * h * y) - y + 1 - h) / len(y)


class MultiClassLogisticRegression:
    def __init__(self):
        self.hypothesis_list = []
        self.classes_amount = 0
        self.m = 0

    def fit(self, X, y, eta=0.1, iterations=256):
        self.classes_amount = len(set(y))
        self.m = len(y)
        for i in range(self.classes_amount):
            copy_y = np.array(y, dtype=int)
            for j in range(self.m):
                if copy_y[j] == i:
                    copy_y[j] = 1
                else:
                    copy_y[j] = 0
            model = LogisticRegression()
            model.fit(X, copy_y, eta, iterations)
            self.hypothesis_list.append(model)

    def predict(self, X):
        m = len(X)
        predictions_mat = np.zeros(shape=(self.classes_amount, m), dtype=float)
        for i, model in enumerate(self.hypothesis_list):
            predictions_mat[i] = model.predict(X)
        predictions = [0] * m
        for i in range(m):
            predictions[i] = np.argmax(predictions_mat[:, i])
        return np.array(predictions, dtype=int)
