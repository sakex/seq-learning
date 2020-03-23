import scipy
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel

DEBUG = True  # Uncomment this line for debug purposes

if DEBUG:
    import sys
    import os

    sys.path.append(os.path.abspath('Debug'))
import sequential_learning as sl

n = 400
n_var = 2
noise = .16

real_c = np.full((n_var, n_var), 1 / 8 * (3 + 2 * np.cos(2) - np.cos(4)))


kernel = ConstantKernel(1e-8) * RBF(length_scale=np.array([1.0, 1.0])) + WhiteKernel(noise_level=1)
# kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-10)

design = np.random.uniform(size=n * n_var).reshape((-1, n_var))

response = np.array([np.sin(np.sum(row)) for row in design])
gpr.fit(design, response)

L_inv = scipy.linalg.solve_triangular(gpr.L_.T, np.eye(gpr.L_.shape[0]))
k_inv = L_inv.dot(L_inv.T)
sigma_f = gpr.kernel_.k1.get_params()['k1__constant_value']

theta = gpr.kernel_.k1.get_params()['k2__length_scale']

mat = sl.C_gp(design, response, theta, k_inv*sigma_f, "RBF")
print(mat)
norm = np.linalg.norm(mat - real_c)
print(norm)
