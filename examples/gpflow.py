import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

DEBUG = True  # Uncomment this line for debug purposes

if DEBUG:
    import sys
    import os

    sys.path.append(os.path.abspath('Debug'))
import sequential_learning as sl


n = 400
n_var = 2
real_c = np.full((2, 2), 1 / 8 * (3 + 2 * np.cos(2) - np.cos(4)))
design = np.random.uniform(size=n * n_var).reshape(-1, 2)
response = np.apply_along_axis(lambda x: np.sin(np.sum(x)), 1, design)

kernel = RBF(length_scale=(1, 1)) + WhiteKernel(noise_level=.5)
# The only way to use this is to look at their code, no explanation anywhere in the docs
# Spent hours on this
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                               optimizer="fmin_l_bfgs_b")
gpr.fit(design, response.reshape(-1, 1))

theta = gpr.kernel_.k1.theta

print(theta)

cov = gpr.kernel_.__call__(design)
k_inv = np.linalg.inv(cov + np.eye(n) * gpr.kernel_.k2.noise_level)
print(k_inv)

mat = sl.C_gp(design, response, theta, k_inv)
print(real_c)
print()
print(mat)
