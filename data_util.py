import numpy as np
import pandas as pd

#Generate artificial polinomial data
def poly_data_generator(N=50):
    err=0.8
    rseed=1
    # randomly sample the data
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)

    return X, y
