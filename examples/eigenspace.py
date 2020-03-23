import scipy
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


DEBUG = True  # Uncomment this line for debug purposes

if DEBUG:
    import sys
    import os

    sys.path.append(os.path.abspath('Debug'))
import sequential_learning as sl


def c_consistence():
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
    norm = np.linalg.norm(mat - real_c)
    assert norm < 1e-3


def subspace_dist(A, B):
    return np.linalg.norm(np.dot(A, A.T) - np.dot(B, B.T), ord=2)


def test_precomputed():
    from data import design, response, kinv, theta
    response, theta = map(lambda item: np.array(item), (response, theta))
    kinv = np.array(kinv).reshape(400, 400, order="F")
    design = np.array(design).reshape(-1, 2, order="F")
    true_sub = np.array((1, 1)) / np.sqrt(2)
    mat = sl.C_gp(design, response, theta, kinv, "Gaussian")
    sub_est = np.linalg.eig(mat)[1][:, 0]
    print(mat)
    assert subspace_dist(sub_est.reshape(1, 2), true_sub.reshape(1, 2)) < 1e-5


if __name__ == "__main__":
    c_consistence()
    # test_precomputed()
