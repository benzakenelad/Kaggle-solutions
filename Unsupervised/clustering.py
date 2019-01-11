import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math as m
from sklearn.mixture import gaussian_mixture
import time

'''
test data
'''
l = list()
for i in range(90):
    for j in range(90):
        l.append([i - 40, j - 40])
test_data = np.array(l)
'''
end
'''

n = 500

miu1 = [0, 10]
cov1 = [[1, 1], [1, 25]]
miu2 = [30, 30]
cov2 = [[4, 4], [4, 16]]
miu3 = [20, -10]
cov3 = [[4, 9], [9, 25]]

data1 = np.random.multivariate_normal(miu1, cov1, size=n)
data2 = np.random.multivariate_normal(miu2, cov2, size=n)
data3 = np.random.multivariate_normal(miu3, cov3, size=n)

data = np.concatenate((data1, data2, data3))
np.random.shuffle(data)

print(data.shape)

plt.xlim(-40, 50)
plt.ylim(-40, 50)

start_time = time.time()

gmm = gaussian_mixture.GaussianMixture(n_components=3)
gmm.fit(data)

data_labels = gmm.predict(data)
test_labels = gmm.predict(test_data)

end_time = time.time()

print((end_time - start_time))

f = plt.figure(1)
plt.scatter(data[:, 0], data[:, 1], c=data_labels, s=20, cmap='viridis');


# plt.plot(data[:, 0], data[:, 1], marker="o", linestyle="")
# plt.show()

g = plt.figure(2)
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, s=20, cmap='viridis');


plt.show()
