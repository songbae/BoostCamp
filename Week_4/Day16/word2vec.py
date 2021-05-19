import numpy as np

c = np.array([[1, 0, 0, 0, 0, 0, 0]])
W = np.random.randn(7, 3)
h = np.matmul(c, W)
print(c.shape)
print(W.shape)
print(h)
import sys
sys.path.append("..")
from comman.layers import MatMul
