import numpy as np
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

DEBUG = True  # Uncomment this line for debug purposes

if DEBUG:
    import sys
    import os

    sys.path.append(os.path.abspath('Debug'))
import sequential_learning as sl


def eigenspace_consistence():
    x, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(x, y)
    theta = gpr.kernel_.theta
    print(sl.C_gp(x, y, theta))


if __name__ == "__main__":
    eigenspace_consistence()
