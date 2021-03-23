from TagNN import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

P = np.logspace(1, 8, 200) # in dyne/cm^2
T = TlayersNN(P, 1000, 1e5) # in K
Pbar = P/1e6 # in bar

plt.semilogy(T, Pbar)
plt.gca().invert_yaxis()
plt.show()



