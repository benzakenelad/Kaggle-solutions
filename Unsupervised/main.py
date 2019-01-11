import numpy as np
import matplotlib.pyplot as plt

n = 100

x = np.arange(n)
delta = np.random.uniform(-10, 10, size=(n,))
y = .4 * x + 3 + delta

m_of_derivative_by_m = float(np.sum(x * x))  # sigma(Xi^2)
b_of_derivative_by_m = float(np.sum(x))  # sigma(Xi)
f_of_derivative_by_m = float(np.sum((-x) * y))  # -sigma(Xi*Yi)

m_of_derivative_by_b = float(np.sum(x))
b_of_derivative_by_b = float(n)
f_of_derivative_by_b = float(np.sum(-y))  # -sigma(Yi)

# stage 1

m1 = m_of_derivative_by_m / b_of_derivative_by_m
f1 = f_of_derivative_by_m / b_of_derivative_by_m
m2 = m_of_derivative_by_b / b_of_derivative_by_b
f2 = f_of_derivative_by_b / b_of_derivative_by_b

m = m1 - m2
f = f1 - f2
real_m = (-f) / m

m = (real_m * m_of_derivative_by_b) / b_of_derivative_by_b
f = f_of_derivative_by_b / b_of_derivative_by_b
real_b = -m - f

print(real_m)
print(real_b)

plt.plot(x, y, marker="o", linestyle="")

line_y = (real_m * np.array([1,100])) + real_b

plt.plot(np.array([1,100]), line_y, linestyle='solid')
plt.show()
