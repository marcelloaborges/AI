import numpy as np

# rewards           array([0.00166667, 0.00166667,
# rewards_future    array([0.01287588, 0.01287588,

rewards = np.array( [0.00166667, 0.001866] )
a =  rewards[::-1]
b = a.cumsum(axis=0)
c = b[::-1]
rewards_future = np.array( [0.01287588, 0.01287588] )