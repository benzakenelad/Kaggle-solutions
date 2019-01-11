import numpy as np

N = 10000
b = np.random.random_integers(-20, 20, size=(N, N))
b_symm = (b + b.T) / 2

print(b_symm)
w, v = np.linalg.eig(b_symm)
print(w)
