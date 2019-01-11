import numpy as np

A = np.array(
    [[9, 13, 2, - 1, 2], [-3, - 2, 11, - 7, 8], [11, - 3, 8, - 11, - 11], [9, - 2, - 12, 1, 3], [0, 1, - 10, 0, - 12]])
b = np.array([8, -2, 4, 88, 22])

for i in range(500):
    b = np.dot(A, b) / np.linalg.norm(b)

print("eigen vector : {}".format(b / b[4]))
print("eigenvalue : {}".format(A.dot(b)[0] / b[0]))
