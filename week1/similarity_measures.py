import numpy as np
import math

# Euclidean distance
def euclid_dist(vec1, vec2):
    d = np.sqrt(np.sum([(a - b) ** 2 for (a, b) in zip(vec1, vec2)]))
    return d

# L1 distance
def L1_dist(vec1, vec2):
    d = np.sum([abs(a + b) for (a, b) in zip(vec1, vec2)])
    return d

# Chi square distance
def chiSquare_dist(vec1, vec2):
    d = np.sum([((a - b) ** 2) / (a + b) for (a, b) in zip(vec1, vec2)])
    return d

# Histogram intersection (similarity)
def Hist_intersection(vec1, vec2):
    i = np.sum([min(a, b) for (a, b) in zip(vec1, vec2)])
    return i

# Hellinger kernel (similarity)
def Hellinger_kernel(vec1, vec2):
    k = np.sum([math.sqrt(a * b) for (a, b) in zip(vec1, vec2)])
    return k