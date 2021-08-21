import numpy as np


def compute_normal(triangle):
    assert triangle.shape[0] == 3
    v1, v2, v3 = triangle

    A = v2 - v1
    B = v3 - v1

    N = np.zeros(3)

    N[0] = A[1] * B[2] - A[2] * B[1]
    N[1] = A[2] * B[0] - A[0] * B[2]
    N[2] = A[0] * B[1] - A[1] * B[0]

    norm = np.sqrt(N[0] ** 2 + N[1] ** 2 + N[2] ** 2)

    return N / norm
