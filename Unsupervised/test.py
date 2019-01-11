import time

import numpy as np
import matplotlib.pyplot as plt

l = list()
for i in range(90):
    for j in range(90):
        l.append([i-40,j-40])
test_data = np.array(l)

plt.plot(test_data[:, 0], test_data[:, 1], marker="o", linestyle="")
plt.show()
