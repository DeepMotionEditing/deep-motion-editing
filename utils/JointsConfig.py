import numpy as np

needed_joints = np.array([
    0,
    2,  3,  4,  5,
    7,  8,  9, 10,
    12, 13, 15, 16,
    18, 19, 20, 22,
    25, 26, 27, 29])


J_all = 31
J_needed = 21


J_parents = np.array([
    -1,
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    10, 13, 14, 15,
    10, 17, 18, 19
])
