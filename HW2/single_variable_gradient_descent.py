import numpy as np
import matplotlib.pyplot as plt

alpha = 0.01 
x = np.arange(-1, 1, 0.01)
y = x**2 

# y_prime = 2*x

curr_x = np.random.uniform(-1, 1)
curr_y = curr_x**2

slope = curr_x * 2

while abs(slope) > 0.000000001:
  curr_x -= alpha * slope 
  curr_y = curr_x**2
  slope = curr_x * 2
  plt.scatter(curr_x, curr_y, edgecolors="red")

plt.scatter(curr_x, curr_y)
plt.plot(x, y)
plt.show()