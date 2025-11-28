import numpy as np

from beliefPropagation import performBeliefPropagation
from drawUtils import plotMatrix

path = "steane"
# path = "[[72, 12, 6]]"
# path = "[[90, 8, 10]]"
# path = "[[108, 8, 10]]"
# path = "[[144, 12, 12]]"
# path = "[[288, 12, 18]]"

H = np.load(f'codes/{path}.npz')['Hx']

plotMatrix(H, path=path)
p = 0.1
initialBelief = [np.log((1-p)/p)] * len(H[0])
error = np.zeros(len(H[0]), dtype=int)

error[0] = 1
error[1] = 1
# error[2] = 1

print(f"Error introduced: {error}")

detection, isSyndromeFound = performBeliefPropagation(H, error, initialBelief)

print(detection, isSyndromeFound)