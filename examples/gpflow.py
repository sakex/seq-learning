import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

DEBUG = True  # Uncomment this line for debug purposes

if DEBUG:
    import sys
    import os

    sys.path.append(os.path.abspath('Debug'))
import sequential_learning as sl


n = 400
n_var = 2
noise = .4

rbf = ConstantKernel(1.0) * RBF(length_scale=np.array([1.0, 1.0]))
gpr = GaussianProcessRegressor(kernel=rbf, alpha=noise**2)

design = np.random.uniform(size=n * n_var).reshape((-1, n_var))

response = np.array([np.sin(np.sum(row)) for row in design])
# Reuse training data from previous 1D example
gpr.fit(design, response)

# Compute posterior predictive mean and covariance
mu_s, cov_s = gpr.predict(design, return_cov=True)

# Obtain optimized kernel parameters
l = gpr.kernel_.k2.get_params()['length_scale']
sigma_f = np.sqrt(gpr.kernel_.k1.get_params()['constant_value'])

# Compare with previous results
print(gpr)
print(l)
print(sigma_f)
k = gpr.kernel_.__call__(design)
scale = k[0][0]
k_inv = np.linalg.inv(cov_s)
theta = l

mat = sl.C_gp(design, response, theta, k_inv, "Matern52")
print(mat)
