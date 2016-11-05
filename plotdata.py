# read and plot
import numpy as np 
import matplotlib.pyplot as plt

data = np.loadtxt('example.txt')
# data = np.random.random((408, 1600))
plt.imshow(data, cmap='hot', interpolation='nearest')
plt.show()