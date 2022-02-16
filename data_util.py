import numpy as np
import pandas as pd

#Generate artificial polinomial data
def poly_data_generator():
    def make_data(N=30, err=0.8, rseed=1):
    # randomly sample the data
        rng = np.random.RandomState(rseed)
        X = rng.rand(N, 1) ** 2
        y = 10 - 1. / (X.ravel() + 0.1)
        if err > 0:
            y += err * rng.randn(N)
        return X, y
    X1, y1 = make_data(N=100, rseed = 0)
    X2, y2 = make_data(N=20, rseed = 42)

    train_data = pd.DataFrame({'X':X1.ravel(), 'y':y1})
    test_data = pd.DataFrame({'X':X2.ravel(), 'y':y2})

    X_train = train_data[["X"]]
    y_train = train_data["y"]

    X_test = test_data[["X"]]
    y_test = test_data["y"]
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
