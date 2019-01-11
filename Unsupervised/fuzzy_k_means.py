import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math as m
from sklearn.mixture import gaussian_mixture
import time
from sklearn_extensions.fuzzy_kmeans import KMedians, FuzzyKMeans, KMeans

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

fuzzy = FuzzyKMeans(k = 3)
fuzzy.fit(data)

centers = fuzzy.cluster_centers_

end_time = time.time()

print((end_time - start_time))


plt.plot(data[:, 0], data[:, 1], marker="o", linestyle="")
plt.plot(centers[:, 0], centers[:, 1], marker="o", linestyle="",color='red')
plt.show()


