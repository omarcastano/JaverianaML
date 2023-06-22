import numpy as np
import pandas as pd


def poly_data_generator(N: int = 50):
    """
    Generate artificial polynomial data

    Arguments:
    ----------
    N: int, default=50
        Number of samples

    Returns:
    --------
    X: numpy.ndarray
        Input data
    y: numpy.ndarray
        Output data

    Examples:
    ---------
    >>> X, y = poly_data_generator(N=50)
    >>> print(X.shape)
    (50, 1)
    """

    err = 0.8
    rseed = 1
    # randomly sample the data
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1.0 / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)

    return X, y
