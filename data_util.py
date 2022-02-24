import numpy as np
import pandas as pd

#Generate artificial polinomial data
def poly_data_generator(n=50):
 
    def make_data(N=N, err=0.8, rseed=1):
    # randomly sample the data
        rng = np.random.RandomState(rseed)
        X = rng.rand(N, 1) ** 2
        y = 10 - 1. / (X.ravel() + 0.1)
        if err > 0:
            y += err * rng.randn(N)
        return X, y
    X, y = make_data(N=n, rseed = 0)

    return X, y
