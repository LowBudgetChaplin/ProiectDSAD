import numpy as np


def standardizare(X):
    medii = np.mean(a=X, axis=0)
    abateriStd = np.std(a=X, axis=0)
    return (X - medii) / abateriStd
